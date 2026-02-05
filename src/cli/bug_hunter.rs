//! Bug Hunter CLI Module
//!
//! Command-line interface for proactive bug hunting.
//!
//! Implements BH-01 to BH-15 from the Popperian Falsification Checklist.

use crate::ansi_colors::Colorize;
use crate::bug_hunter::{
    hunt, hunt_ensemble, hunt_with_spec, hunt_with_ticket, CrashBucketingMode, DefectCategory,
    Finding, FindingSeverity, HuntConfig, HuntMode, HuntResult, HuntStats, LocalizationStrategy,
    SbflFormula,
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

        /// Enable PMAT quality-weighted suspiciousness (BH-21)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// Use PMAT to scope targets by quality (BH-22)
        #[arg(long)]
        pmat_scope: bool,

        /// PMAT query string for scoping (BH-22)
        #[arg(long)]
        pmat_query: Option<String>,

        /// Coverage file (lcov format) for hotpath weighting
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// Coverage weight factor for hotpath weighting (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        coverage_weight: f64,
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

        /// Fault localization strategy (BH-16 to BH-19)
        #[arg(long, value_enum, default_value = "sbfl")]
        strategy: LocalizationStrategyArg,

        /// Crash bucketing mode (BH-20)
        #[arg(long, value_enum, default_value = "none")]
        crash_bucket: CrashBucketArg,

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

        /// Enable PMAT quality-weighted suspiciousness (BH-21)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// PMAT query string for scoping (BH-22)
        #[arg(long)]
        pmat_query: Option<String>,
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

        /// Enable PMAT quality-weighted suspiciousness (BH-21/BH-25)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// PMAT query string for scoping
        #[arg(long)]
        pmat_query: Option<String>,
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

    /// Cross-stack bug analysis for Sovereign AI Stack
    ///
    /// Scans trueno, aprender, realizar, entrenar, repartir in parallel
    /// and generates a consolidated cross-stack report.
    Stack {
        /// Base directory containing stack crates (default: parent of cwd)
        #[arg(long)]
        base: Option<PathBuf>,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Crates to scan (default: trueno,aprender,realizar,entrenar,repartir)
        #[arg(long, value_delimiter = ',')]
        crates: Option<Vec<String>>,

        /// Generate GitHub issue body
        #[arg(long)]
        issue: bool,
    },

    /// Show only new findings compared to a baseline
    ///
    /// Compares current findings against a git branch or time period.
    Diff {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Base git branch/commit to compare against (e.g., "main", "HEAD~5")
        #[arg(long)]
        base: Option<String>,

        /// Time period to compare against (e.g., "7d", "2w", "1m")
        #[arg(long)]
        since: Option<String>,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Save current findings as new baseline
        #[arg(long)]
        save_baseline: bool,
    },

    /// Show tech debt trends over time
    Trend {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Number of weeks to show
        #[arg(long, default_value = "12")]
        weeks: usize,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Group related findings by root cause
    Triage {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
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

/// Localization strategy CLI argument (BH-16 to BH-19).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LocalizationStrategyArg {
    /// Spectrum-Based Fault Localization only
    #[default]
    Sbfl,
    /// Mutation-Based Fault Localization (BH-16)
    Mbfl,
    /// Causal inference (BH-17)
    Causal,
    /// Multi-channel combination (BH-19)
    MultiChannel,
    /// Hybrid SBFL + MBFL
    Hybrid,
}

impl From<LocalizationStrategyArg> for LocalizationStrategy {
    fn from(arg: LocalizationStrategyArg) -> Self {
        match arg {
            LocalizationStrategyArg::Sbfl => LocalizationStrategy::Sbfl,
            LocalizationStrategyArg::Mbfl => LocalizationStrategy::Mbfl,
            LocalizationStrategyArg::Causal => LocalizationStrategy::Causal,
            LocalizationStrategyArg::MultiChannel => LocalizationStrategy::MultiChannel,
            LocalizationStrategyArg::Hybrid => LocalizationStrategy::Hybrid,
        }
    }
}

/// Crash bucketing mode CLI argument (BH-20).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum CrashBucketArg {
    /// No bucketing
    #[default]
    None,
    /// Stack trace similarity
    StackTrace,
    /// Semantic root cause analysis
    Semantic,
}

impl From<CrashBucketArg> for CrashBucketingMode {
    fn from(arg: CrashBucketArg) -> Self {
        match arg {
            CrashBucketArg::None => CrashBucketingMode::None,
            CrashBucketArg::StackTrace => CrashBucketingMode::StackTrace,
            CrashBucketArg::Semantic => CrashBucketingMode::Semantic,
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
            pmat_quality,
            quality_weight,
            pmat_scope,
            pmat_query,
            coverage,
            coverage_weight,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Analyze,
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                max_findings,
                llm_filter,
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_scope,
                pmat_query,
                coverage_path: coverage,
                coverage_weight,
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
            strategy,
            crash_bucket,
            top_n,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Hunt,
                targets: vec![PathBuf::from("src")],
                max_findings: top_n,
                sbfl_formula: formula.into(),
                coverage_path: coverage,
                localization_strategy: strategy.into(),
                crash_bucketing: crash_bucket.into(),
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
            pmat_quality,
            quality_weight,
            pmat_query,
        } => {
            let config = HuntConfig {
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_query,
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
            pmat_quality,
            quality_weight,
            pmat_query,
        } => {
            let config = HuntConfig {
                min_suspiciousness,
                // Quick mode does pattern-only scan, no clippy/coverage
                mode: if quick { HuntMode::Quick } else { HuntMode::Analyze },
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_query,
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

        BugHunterCommand::Stack {
            base,
            min_suspiciousness,
            format,
            crates,
            issue,
        } => {
            handle_stack_command(base, min_suspiciousness, format, crates, issue)
        }

        BugHunterCommand::Diff {
            path,
            base,
            since,
            min_suspiciousness,
            format,
            save_baseline,
        } => {
            handle_diff_command(path, base, since, min_suspiciousness, format, save_baseline)
        }

        BugHunterCommand::Trend {
            path,
            weeks,
            format,
        } => {
            handle_trend_command(path, weeks, format)
        }

        BugHunterCommand::Triage {
            path,
            min_suspiciousness,
            format,
        } => {
            handle_triage_command(path, min_suspiciousness, format)
        }
    }
}

/// Handle the cross-stack bug analysis command.
fn handle_stack_command(
    base: Option<PathBuf>,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
    crates: Option<Vec<String>>,
    generate_issue: bool,
) -> Result<(), String> {
    use std::collections::HashMap;

    // Default stack crates
    let default_crates = vec![
        "trueno".to_string(),
        "aprender".to_string(),
        "realizar".to_string(),
        "entrenar".to_string(),
        "repartir".to_string(),
    ];
    let crate_list = crates.unwrap_or(default_crates);

    // Determine base directory
    let base_dir = base.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from(".."))
    });

    eprintln!(
        "{}",
        format!("Scanning {} crates in {}...", crate_list.len(), base_dir.display()).dimmed()
    );

    // Scan each crate in parallel
    let results: Vec<(String, HuntResult)> = std::thread::scope(|s| {
        let handles: Vec<_> = crate_list
            .iter()
            .map(|crate_name| {
                let base = &base_dir;
                let min_susp = min_suspiciousness;
                s.spawn(move || {
                    let crate_path = base.join(crate_name);
                    if !crate_path.exists() {
                        eprintln!("  {} {} (not found)", "[SKIP]".yellow(), crate_name);
                        return None;
                    }
                    eprintln!("  {} {}...", "[SCAN]".dimmed(), crate_name);
                    let config = HuntConfig {
                        mode: HuntMode::Analyze,
                        min_suspiciousness: min_susp,
                        ..Default::default()
                    };
                    let result = hunt(&crate_path, config);
                    Some((crate_name.clone(), result))
                })
            })
            .collect();

        handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .collect()
    });

    if results.is_empty() {
        return Err("No crates found to analyze".to_string());
    }

    // Aggregate stats
    let mut total_findings = 0usize;
    let mut total_critical = 0usize;
    let mut total_high = 0usize;
    let mut by_category: HashMap<DefectCategory, usize> = HashMap::new();
    let mut crate_stats: Vec<CrateStats> = Vec::new();

    for (crate_name, result) in &results {
        let stats = &result.stats;
        total_findings += stats.total_findings;

        let critical = stats
            .by_severity
            .get(&FindingSeverity::Critical)
            .copied()
            .unwrap_or(0);
        let high = stats
            .by_severity
            .get(&FindingSeverity::High)
            .copied()
            .unwrap_or(0);
        total_critical += critical;
        total_high += high;

        for (cat, count) in &stats.by_category {
            *by_category.entry(*cat).or_insert(0) += count;
        }

        crate_stats.push(CrateStats {
            name: crate_name.clone(),
            total: stats.total_findings,
            critical,
            high,
            gpu: stats
                .by_category
                .get(&DefectCategory::GpuKernelBugs)
                .copied()
                .unwrap_or(0),
            debt: stats
                .by_category
                .get(&DefectCategory::HiddenDebt)
                .copied()
                .unwrap_or(0),
            test: stats
                .by_category
                .get(&DefectCategory::TestDebt)
                .copied()
                .unwrap_or(0),
            silent: stats
                .by_category
                .get(&DefectCategory::SilentDegradation)
                .copied()
                .unwrap_or(0),
            memory: stats
                .by_category
                .get(&DefectCategory::MemorySafety)
                .copied()
                .unwrap_or(0),
        });
    }

    match format {
        BugHunterOutputFormat::Json => {
            output_stack_json(&crate_stats, &results);
        }
        BugHunterOutputFormat::Markdown | BugHunterOutputFormat::Text => {
            output_stack_text(&crate_stats, total_findings, total_critical, total_high, &by_category);
            if generate_issue {
                println!();
                output_stack_issue(&crate_stats, &results);
            }
        }
        BugHunterOutputFormat::Sarif => {
            // Combine all findings into one SARIF output
            let mut all_findings: Vec<Finding> = Vec::new();
            for (_, result) in &results {
                all_findings.extend(result.findings.clone());
            }
            if let Some((_, first_result)) = results.first() {
                let mut combined = first_result.clone();
                combined.findings = all_findings;
                output_sarif(&combined);
            }
        }
    }

    Ok(())
}

/// Stats for a single crate in the stack analysis.
#[derive(Debug)]
struct CrateStats {
    name: String,
    total: usize,
    critical: usize,
    high: usize,
    gpu: usize,
    debt: usize,
    test: usize,
    silent: usize,
    memory: usize,
}

/// Output stack analysis as text.
fn output_stack_text(
    crate_stats: &[CrateStats],
    total_findings: usize,
    total_critical: usize,
    total_high: usize,
    by_category: &std::collections::HashMap<DefectCategory, usize>,
) {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
            .bright_cyan()
    );
    println!(
        "{}",
        "║           CROSS-STACK BUG ANALYSIS - SOVEREIGN AI STACK               ║"
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
            .bright_cyan()
    );
    println!();

    println!(
        "{}",
        "┌─────────────────────────────────────────────────────────────────────────┐".dimmed()
    );
    println!(
        "{}",
        "│ STACK DEPENDENCY CHAIN: trueno → aprender → realizar → entrenar        │".dimmed()
    );
    println!(
        "{}",
        "└─────────────────────────────────────────────────────────────────────────┘".dimmed()
    );
    println!();

    println!("{}", "SUMMARY BY CRATE:".bold());
    println!("┌──────────────┬────────┬──────────┬──────┬────────┬──────┬────────┬──────┐");
    println!("│ Crate        │ Total  │ Critical │ High │ GPU    │ Debt │ Test   │ Mem  │");
    println!("├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┤");

    for stats in crate_stats {
        println!(
            "│ {:<12} │ {:>6} │ {:>8} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>4} │",
            stats.name,
            stats.total,
            stats.critical,
            stats.high,
            stats.gpu,
            stats.debt,
            stats.test,
            stats.memory
        );
    }

    println!("├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┤");
    println!(
        "│ {:<12} │ {:>6} │ {:>8} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>4} │",
        "TOTAL".bold(),
        total_findings,
        total_critical,
        total_high,
        by_category.get(&DefectCategory::GpuKernelBugs).unwrap_or(&0),
        by_category.get(&DefectCategory::HiddenDebt).unwrap_or(&0),
        by_category.get(&DefectCategory::TestDebt).unwrap_or(&0),
        by_category.get(&DefectCategory::MemorySafety).unwrap_or(&0)
    );
    println!("└──────────────┴────────┴──────────┴──────┴────────┴──────┴────────┴──────┘");
    println!();

    // Risk summary
    println!("{}", "CROSS-STACK INTEGRATION RISKS:".bold());
    println!();

    let gpu_total = by_category.get(&DefectCategory::GpuKernelBugs).unwrap_or(&0);
    if *gpu_total > 0 {
        println!(
            "  {} GPU Kernel Chain (trueno SIMD → realizar CUDA):",
            "1.".yellow()
        );
        println!("     • {} GPU kernel bugs detected", gpu_total);
        println!("     • Impact: Potential performance degradation or kernel failures");
        println!();
    }

    let debt_total = by_category.get(&DefectCategory::HiddenDebt).unwrap_or(&0);
    if *debt_total > 0 {
        println!(
            "  {} Hidden Technical Debt:",
            "2.".yellow()
        );
        println!("     • {} euphemism patterns (placeholder, stub, etc.)", debt_total);
        println!("     • Impact: Incomplete implementations may cause failures");
        println!();
    }

    let test_total = by_category.get(&DefectCategory::TestDebt).unwrap_or(&0);
    if *test_total > 0 {
        println!(
            "  {} Test Debt:",
            "3.".yellow()
        );
        println!("     • {} tests ignored or removed", test_total);
        println!("     • Impact: Known bugs not being caught by CI");
        println!();
    }
}

/// Output stack analysis as JSON.
fn output_stack_json(crate_stats: &[CrateStats], results: &[(String, HuntResult)]) {
    use serde_json::json;

    let crates: Vec<_> = crate_stats
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "total": s.total,
                "critical": s.critical,
                "high": s.high,
                "gpu": s.gpu,
                "debt": s.debt,
                "test": s.test,
                "silent": s.silent,
                "memory": s.memory
            })
        })
        .collect();

    let all_findings: Vec<_> = results
        .iter()
        .flat_map(|(name, r)| {
            r.findings.iter().map(move |f| {
                json!({
                    "crate": name,
                    "file": f.file,
                    "line": f.line,
                    "severity": format!("{:?}", f.severity),
                    "category": format!("{:?}", f.category),
                    "title": f.title,
                    "suspiciousness": f.suspiciousness
                })
            })
        })
        .collect();

    let output = json!({
        "crates": crates,
        "findings": all_findings,
        "totals": {
            "findings": crate_stats.iter().map(|s| s.total).sum::<usize>(),
            "critical": crate_stats.iter().map(|s| s.critical).sum::<usize>(),
            "high": crate_stats.iter().map(|s| s.high).sum::<usize>()
        }
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
}

/// Output GitHub issue body for cross-stack report.
fn output_stack_issue(crate_stats: &[CrateStats], results: &[(String, HuntResult)]) {
    println!("{}", "--- GITHUB ISSUE BODY ---".dimmed());
    println!();
    println!("## Cross-Stack Bug Analysis - Sovereign AI Stack");
    println!();
    println!("### Summary by Crate");
    println!();
    println!("| Crate | Total | Critical | High | GPU | Debt | Test | Mem |");
    println!("|-------|-------|----------|------|-----|------|------|-----|");

    for s in crate_stats {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            s.name, s.total, s.critical, s.high, s.gpu, s.debt, s.test, s.memory
        );
    }

    let totals: CrateStats = CrateStats {
        name: "**TOTAL**".to_string(),
        total: crate_stats.iter().map(|s| s.total).sum(),
        critical: crate_stats.iter().map(|s| s.critical).sum(),
        high: crate_stats.iter().map(|s| s.high).sum(),
        gpu: crate_stats.iter().map(|s| s.gpu).sum(),
        debt: crate_stats.iter().map(|s| s.debt).sum(),
        test: crate_stats.iter().map(|s| s.test).sum(),
        silent: crate_stats.iter().map(|s| s.silent).sum(),
        memory: crate_stats.iter().map(|s| s.memory).sum(),
    };
    println!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |",
        totals.name,
        totals.total,
        totals.critical,
        totals.high,
        totals.gpu,
        totals.debt,
        totals.test,
        totals.memory
    );

    println!();
    println!("### Critical Findings");
    println!();
    println!("```");
    for (crate_name, result) in results {
        for f in result.findings.iter().filter(|f| matches!(f.severity, FindingSeverity::Critical)) {
            let file_name = f.file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
            println!(
                "{}: {}:{} - {}",
                crate_name, file_name, f.line, f.title
            );
        }
    }
    println!("```");
    println!();
    println!("*Generated by `batuta bug-hunter stack`*");
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

/// Severity badge: [C] bright_red bold, [H] red, [M] yellow, [L] blue, [I] dimmed.
fn severity_badge(severity: &FindingSeverity) -> String {
    match severity {
        FindingSeverity::Critical => format!("{}", "[C]".bright_red().bold()),
        FindingSeverity::High => format!("{}", "[H]".red()),
        FindingSeverity::Medium => format!("{}", "[M]".yellow()),
        FindingSeverity::Low => format!("{}", "[L]".blue()),
        FindingSeverity::Info => format!("{}", "[I]".dimmed()),
    }
}

/// Suspiciousness bar: filled/empty blocks proportional to score.
fn suspiciousness_bar(score: f64, width: usize) -> String {
    let filled = (score * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{} {:.2}",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(empty),
        score
    )
}

/// Compact severity one-liner: "2C 5H 8M 0L 0I".
fn severity_summary_line(stats: &HuntStats) -> String {
    let c = stats
        .by_severity
        .get(&FindingSeverity::Critical)
        .unwrap_or(&0);
    let h = stats
        .by_severity
        .get(&FindingSeverity::High)
        .unwrap_or(&0);
    let m = stats
        .by_severity
        .get(&FindingSeverity::Medium)
        .unwrap_or(&0);
    let l = stats.by_severity.get(&FindingSeverity::Low).unwrap_or(&0);
    let i = stats
        .by_severity
        .get(&FindingSeverity::Info)
        .unwrap_or(&0);
    format!(
        "{}C {}H {}M {}L {}I",
        format!("{}", c).bright_red(),
        format!("{}", h).red(),
        format!("{}", m).yellow(),
        format!("{}", l).blue(),
        format!("{}", i).dimmed(),
    )
}

/// Print category distribution with horizontal bar chart.
fn print_category_distribution(stats: &HuntStats) {
    if stats.by_category.is_empty() {
        return;
    }
    println!("{}", "Category Distribution:".bold());
    let max_count = stats.by_category.values().max().copied().unwrap_or(1);
    let bar_width = 20;
    let mut sorted: Vec<_> = stats.by_category.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in &sorted {
        let bar_len = if max_count > 0 {
            (**count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };
        println!(
            "  {:<22} {} {}",
            format!("{}", cat),
            "\u{2588}".repeat(bar_len),
            count
        );
    }
    println!();
}

/// Print top-N hotspot files by finding count.
fn print_hotspot_files(findings: &[Finding], n: usize) {
    if findings.is_empty() {
        return;
    }
    let mut file_counts: std::collections::HashMap<&std::path::Path, usize> =
        std::collections::HashMap::new();
    for f in findings {
        *file_counts.entry(&f.file).or_default() += 1;
    }
    let mut sorted: Vec<_> = file_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top = &sorted[..sorted.len().min(n)];
    if top.is_empty() {
        return;
    }
    let max_count = top[0].1;
    let bar_width = 15;

    println!("{}", "Hotspot Files:".bold());
    for (file, count) in top {
        let bar_len = if max_count > 0 {
            (*count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };
        println!(
            "  {:<40} {} {}",
            format!("{}", file.display()).dimmed(),
            "\u{2588}".repeat(bar_len),
            count
        );
    }
    println!();
}

/// Output as human-readable text.
fn output_text(result: &HuntResult) {
    let sep = "\u{2500}".repeat(74);

    // Header
    println!("\n{}", "Bug Hunter Report".bright_cyan().bold());
    println!("{}", sep.dimmed());
    println!(
        "Mode: {}  Findings: {}  Duration: {}ms",
        format!("{}", result.mode).bright_yellow(),
        result.findings.len(),
        result.duration_ms,
    );

    // Phase timings (if any nonzero)
    let pt = &result.phase_timings;
    if pt.mode_dispatch_ms > 0 || pt.pmat_index_ms > 0 || pt.finalize_ms > 0 {
        let mut parts = Vec::new();
        if pt.mode_dispatch_ms > 0 {
            parts.push(format!("scan={}ms", pt.mode_dispatch_ms));
        }
        if pt.pmat_index_ms > 0 {
            parts.push(format!("pmat-index={}ms", pt.pmat_index_ms));
        }
        if pt.pmat_weights_ms > 0 {
            parts.push(format!("weights={}ms", pt.pmat_weights_ms));
        }
        if pt.finalize_ms > 0 {
            parts.push(format!("finalize={}ms", pt.finalize_ms));
        }
        println!("{}", parts.join("  ").dimmed());
    }

    // Severity summary
    println!(
        "Severity: {}",
        severity_summary_line(&result.stats),
    );
    println!();

    // Category distribution chart
    print_category_distribution(&result.stats);

    // Hotspot files (top 5)
    print_hotspot_files(&result.findings, 5);

    // Findings list
    let top = result.top_findings(20);
    if top.is_empty() {
        println!("{}", "No findings discovered.".green());
    } else {
        println!("{}", "Findings:".bold());
        println!("{}", sep.dimmed());

        for finding in top {
            let badge = severity_badge(&finding.severity);
            let bar = suspiciousness_bar(finding.suspiciousness, 10);
            println!(
                "{} {} {} {}",
                badge,
                finding.id.dimmed(),
                bar,
                finding.location()
            );
            println!("    {}", finding.title.bright_white());
            if !finding.description.is_empty() {
                println!("    {}", finding.description.dimmed());
            }
            if let Some(risk) = finding.regression_risk {
                println!(
                    "    {}",
                    format!("Regression Risk: {:.2}", risk).yellow()
                );
            }
            // Display git blame info if available
            if let (Some(author), Some(commit), Some(date)) =
                (&finding.blame_author, &finding.blame_commit, &finding.blame_date)
            {
                println!(
                    "    {}",
                    format!("Blame: {} ({}) {}", author, commit, date).dimmed()
                );
            }
        }
    }

    // Summary
    println!("\n{}", sep.dimmed());
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
    println!("| ID | Severity | Category | Score | Risk | Location |");
    println!("|-----|----------|----------|-------|------|----------|");

    for finding in result.top_findings(20) {
        let risk = finding
            .regression_risk
            .map(|r| format!("{:.2}", r))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "| {} | {:?} | {:?} | {:.2} | {} | `{}` |",
            finding.id,
            finding.severity,
            finding.category,
            finding.suspiciousness,
            risk,
            finding.location()
        );
    }

    println!("\n---\n");
    println!("*{}*", result.summary());
}

/// Handle the diff command - show only new findings.
fn handle_diff_command(
    path: PathBuf,
    base: Option<String>,
    since: Option<String>,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
    save_baseline: bool,
) -> Result<(), String> {
    use crate::bug_hunter::diff::{Baseline, DiffResult};

    // Run current analysis
    let config = HuntConfig {
        mode: HuntMode::Analyze,
        min_suspiciousness,
        ..Default::default()
    };
    let result = hunt(&path, config);

    // Save baseline if requested
    if save_baseline {
        let baseline = Baseline::from_findings(&result.findings);
        baseline.save(&path)?;
        println!(
            "{}",
            format!("Saved baseline with {} findings", result.findings.len()).green()
        );
        return Ok(());
    }

    // Load existing baseline
    let baseline = Baseline::load(&path);

    if baseline.is_none() && base.is_none() && since.is_none() {
        println!(
            "{}",
            "No baseline found. Run with --save-baseline first, or use --base/--since.".yellow()
        );
        output_result(&result, format);
        return Ok(());
    }

    // Compute diff
    let base_ref = base
        .as_deref()
        .or(since.as_deref())
        .unwrap_or("baseline");

    if let Some(baseline) = baseline {
        let diff = DiffResult::compute(&result, &baseline, base_ref);
        output_diff_result(&diff, format);
    } else {
        // Use git diff for changed files
        let changed = crate::bug_hunter::diff::get_changed_files(
            &path,
            base.as_deref(),
            since.as_deref(),
        );
        let filtered = crate::bug_hunter::diff::filter_changed_files(&result.findings, &changed);

        println!(
            "{}",
            format!(
                "Showing {} findings in {} changed files ({})",
                filtered.len(),
                changed.len(),
                base_ref
            )
            .bright_cyan()
            .bold()
        );
        println!();

        for f in &filtered {
            println!(
                "{} {} {}:{}",
                severity_badge(&f.severity),
                f.id.dimmed(),
                f.file.display(),
                f.line
            );
            println!("    {}", f.title.white());
        }
    }

    Ok(())
}

/// Output diff result.
fn output_diff_result(diff: &crate::bug_hunter::diff::DiffResult, format: BugHunterOutputFormat) {
    match format {
        BugHunterOutputFormat::Json => {
            use serde_json::json;
            let output = json!({
                "new_findings": diff.new_findings.len(),
                "resolved": diff.resolved_count,
                "total_current": diff.total_current,
                "total_baseline": diff.total_baseline,
                "base": diff.base_reference,
                "findings": diff.new_findings.iter().map(|f| json!({
                    "file": f.file,
                    "line": f.line,
                    "severity": format!("{:?}", f.severity),
                    "title": f.title
                })).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
        }
        _ => {
            println!(
                "{}",
                "╔══════════════════════════════════════════════════════════════════════════╗"
                    .bright_cyan()
            );
            println!(
                "{}",
                format!(
                    "║                    DIFF vs {} {:>30}║",
                    diff.base_reference,
                    ""
                )
                .bright_cyan()
                .bold()
            );
            println!(
                "{}",
                "╚══════════════════════════════════════════════════════════════════════════╝"
                    .bright_cyan()
            );
            println!();

            // Summary
            let new_color = if diff.new_findings.is_empty() {
                "0".green()
            } else {
                format!("{}", diff.new_findings.len()).red().bold()
            };
            let resolved_color = if diff.resolved_count == 0 {
                "0".dimmed()
            } else {
                format!("{}", diff.resolved_count).green()
            };

            println!(
                "  {} new findings, {} resolved",
                new_color, resolved_color
            );
            println!(
                "  {} total (was {})",
                diff.total_current.to_string().white(),
                diff.total_baseline.to_string().dimmed()
            );
            println!();

            if diff.new_findings.is_empty() {
                println!("{}", "  No new findings! 🎉".green());
            } else {
                println!("{}", "NEW FINDINGS:".bold());
                println!();
                for f in &diff.new_findings {
                    println!(
                        "{} {} {}:{}",
                        severity_badge(&f.severity),
                        f.id.dimmed(),
                        f.file.display(),
                        f.line
                    );
                    println!("    {}", f.title.white());
                }
            }
        }
    }
}

/// Handle trend command - show tech debt over time.
fn handle_trend_command(
    path: PathBuf,
    weeks: usize,
    format: BugHunterOutputFormat,
) -> Result<(), String> {
    let trend_path = path.join(".pmat").join("bug-hunter-trend.json");

    // Load existing trend data
    let trend_data: Vec<TrendSnapshot> = if trend_path.exists() {
        let content = std::fs::read_to_string(&trend_path)
            .map_err(|e| format!("Failed to read trend data: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Run current analysis and add snapshot
    let config = HuntConfig {
        mode: HuntMode::Analyze,
        min_suspiciousness: 0.3,
        ..Default::default()
    };
    let result = hunt(&path, config);

    let current = TrendSnapshot {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        total: result.findings.len(),
        critical: result
            .stats
            .by_severity
            .get(&FindingSeverity::Critical)
            .copied()
            .unwrap_or(0),
        high: result
            .stats
            .by_severity
            .get(&FindingSeverity::High)
            .copied()
            .unwrap_or(0),
        debt: result
            .stats
            .by_category
            .get(&DefectCategory::HiddenDebt)
            .copied()
            .unwrap_or(0),
    };

    // Save updated trend
    let mut updated_trend = trend_data.clone();
    updated_trend.push(current.clone());
    // Keep only recent snapshots
    if updated_trend.len() > weeks {
        updated_trend = updated_trend.split_off(updated_trend.len() - weeks);
    }
    let pmat_dir = path.join(".pmat");
    std::fs::create_dir_all(&pmat_dir).ok();
    std::fs::write(
        &trend_path,
        serde_json::to_string_pretty(&updated_trend).unwrap_or_default(),
    )
    .ok();

    // Output trend
    match format {
        BugHunterOutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&updated_trend).unwrap_or_default()
            );
        }
        _ => {
            println!("{}", "TECH DEBT TREND".bold());
            println!();
            println!(
                "{:12} {:>8} {:>8} {:>8} {:>8}  {}",
                "Date".dimmed(),
                "Total".dimmed(),
                "Critical".dimmed(),
                "High".dimmed(),
                "Debt".dimmed(),
                "Trend".dimmed()
            );

            let max_total = updated_trend.iter().map(|s| s.total).max().unwrap_or(1);
            for snapshot in &updated_trend {
                let date = chrono::DateTime::from_timestamp(snapshot.timestamp as i64, 0)
                    .map(|d| d.format("%Y-%m-%d").to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                let bar_len = (snapshot.total * 20 / max_total.max(1)).min(20);
                let bar = "━".repeat(bar_len);

                println!(
                    "{:12} {:>8} {:>8} {:>8} {:>8}  {}",
                    date,
                    snapshot.total,
                    snapshot.critical,
                    snapshot.high,
                    snapshot.debt,
                    bar.yellow()
                );
            }
        }
    }

    Ok(())
}

/// Trend snapshot for tracking over time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrendSnapshot {
    timestamp: u64,
    total: usize,
    critical: usize,
    high: usize,
    debt: usize,
}

/// Handle triage command - group related findings.
fn handle_triage_command(
    path: PathBuf,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
) -> Result<(), String> {
    use std::collections::HashMap;

    let config = HuntConfig {
        mode: HuntMode::Analyze,
        min_suspiciousness,
        ..Default::default()
    };
    let result = hunt(&path, config);

    // Group by directory + pattern
    let mut groups: HashMap<String, Vec<&Finding>> = HashMap::new();
    for f in &result.findings {
        let dir = f
            .file
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        // Extract pattern from title
        let pattern = f
            .title
            .strip_prefix("Pattern: ")
            .or_else(|| f.title.strip_prefix("Custom: "))
            .unwrap_or(&f.title);
        let key = format!("{}:{}", dir, pattern);
        groups.entry(key).or_default().push(f);
    }

    // Sort groups by count
    let mut sorted_groups: Vec<_> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    match format {
        BugHunterOutputFormat::Json => {
            use serde_json::json;
            let output: Vec<_> = sorted_groups
                .iter()
                .map(|(key, findings)| {
                    json!({
                        "root_cause": key,
                        "count": findings.len(),
                        "findings": findings.iter().map(|f| json!({
                            "file": f.file,
                            "line": f.line,
                        })).collect::<Vec<_>>()
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
        }
        _ => {
            println!("{}", "AUTO-TRIAGE: Findings by Root Cause".bold());
            println!();

            for (key, findings) in sorted_groups.iter().take(20) {
                let parts: Vec<&str> = key.splitn(2, ':').collect();
                let (dir, pattern) = if parts.len() == 2 {
                    (parts[0], parts[1])
                } else {
                    (".", key.as_str())
                };

                println!(
                    "{} ({} findings)",
                    format!("{}:{}", dir, pattern).white().bold(),
                    findings.len().to_string().yellow()
                );

                for f in findings.iter().take(3) {
                    let file_name = f.file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
                    println!(
                        "  {} {}:{}",
                        severity_badge(&f.severity),
                        file_name.dimmed(),
                        f.line
                    );
                }
                if findings.len() > 3 {
                    println!("  {} more...", format!("... +{}", findings.len() - 3).dimmed());
                }
                println!();
            }
        }
    }

    Ok(())
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
            pmat_quality: false,
            quality_weight: 0.5,
            pmat_scope: false,
            pmat_query: None,
            coverage: None,
            coverage_weight: 0.5,
        };
        // Should not panic
        let _ = handle_bug_hunter_command(cmd);
    }
}
