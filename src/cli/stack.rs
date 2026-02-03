//! Stack command implementations
//!
//! This module contains all stack-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::stack;
use std::io::IsTerminal;
use std::path::PathBuf;

/// Stack output format
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum StackOutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

/// Version bump type
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BumpType {
    Patch,
    Minor,
    Major,
}

/// Stack subcommand
#[derive(Debug, Clone, clap::Subcommand)]
pub enum StackCommand {
    /// Check dependency health across the stack
    Check {
        /// Specific project to check (default: scan workspace)
        #[arg(long)]
        project: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Strict mode (fail on warnings)
        #[arg(long)]
        strict: bool,

        /// Verify against crates.io
        #[arg(long)]
        verify_published: bool,

        /// Offline mode (use cached data)
        #[arg(long)]
        offline: bool,

        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Prepare a stack release
    Release {
        /// Crate to release
        crate_name: Option<String>,

        /// Release all crates
        #[arg(long)]
        all: bool,

        /// Dry run (show what would happen)
        #[arg(long)]
        dry_run: bool,

        /// Version bump type
        #[arg(long, value_enum)]
        bump: Option<BumpType>,

        /// Skip verification
        #[arg(long)]
        no_verify: bool,

        /// Skip confirmation
        #[arg(long, short)]
        yes: bool,

        /// Publish to crates.io
        #[arg(long)]
        publish: bool,
    },

    /// Show stack status
    Status {
        /// Simple text output (no TUI)
        #[arg(long)]
        simple: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Show dependency tree
        #[arg(long)]
        tree: bool,
    },

    /// Sync stack dependencies
    Sync {
        /// Crate to sync
        crate_name: Option<String>,

        /// Sync all crates
        #[arg(long)]
        all: bool,

        /// Dry run
        #[arg(long)]
        dry_run: bool,

        /// Align to specific dependency version
        #[arg(long)]
        align: Option<String>,
    },

    /// Show dependency tree
    Tree {
        /// Output format (ascii, json, dot)
        #[arg(long, default_value = "ascii")]
        format: String,

        /// Include health indicators
        #[arg(long)]
        health: bool,

        /// Filter by layer
        #[arg(long)]
        filter: Option<String>,
    },

    /// Check code quality across stack
    Quality {
        /// Specific component to check
        #[arg(long)]
        component: Option<String>,

        /// Strict mode (fail if not A+)
        #[arg(long)]
        strict: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Verify hero implementation matching
        #[arg(long)]
        verify_hero: bool,

        /// Verbose output
        #[arg(long, short)]
        verbose: bool,

        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Quality gate check for CI/pre-commit
    Gate {
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,

        /// Quiet mode (minimal output)
        #[arg(long, short)]
        quiet: bool,
    },

    /// Check latest versions of PAIML stack crates
    Versions {
        /// Show only outdated crates
        #[arg(long)]
        outdated: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Offline mode (use cached data only)
        #[arg(long)]
        offline: bool,

        /// Include pre-release versions
        #[arg(long)]
        include_prerelease: bool,
    },

    /// Check publish status of stack crates
    PublishStatus {
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,

        /// Clear cache and refresh
        #[arg(long)]
        clear_cache: bool,
    },

    /// Detect version drift in stack dependencies
    Drift {
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Generate fix commands
        #[arg(long)]
        fix: bool,

        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Check cross-project compliance (Makefiles, Cargo.toml, CI)
    Comply {
        /// Specific rule to check (e.g., makefile-targets, cargo-toml-consistency)
        #[arg(long)]
        rule: Option<String>,

        /// Attempt to auto-fix violations
        #[arg(long)]
        fix: bool,

        /// Dry run (show what would be fixed)
        #[arg(long)]
        dry_run: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: ComplyOutputFormat,

        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,

        /// List available rules
        #[arg(long)]
        list_rules: bool,
    },
}

/// Comply output format
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ComplyOutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
    Html,
}

/// Execute read-only stack commands (check, status, tree, versions).
fn dispatch_stack_info(command: StackCommand) -> anyhow::Result<()> {
    match command {
        StackCommand::Check {
            project,
            format,
            strict,
            verify_published,
            offline,
            workspace,
        } => cmd_stack_check(
            project,
            format,
            strict,
            verify_published,
            offline,
            workspace,
        ),
        StackCommand::Status {
            simple,
            format,
            tree,
        } => cmd_stack_status(simple, format, tree),
        StackCommand::Tree {
            format,
            health,
            filter,
        } => cmd_stack_tree(&format, health, filter.as_deref()),
        StackCommand::Versions {
            outdated,
            format,
            offline,
            include_prerelease,
        } => cmd_stack_versions(outdated, format, offline, include_prerelease),
        _ => unreachable!(),
    }
}

/// Execute mutating/quality stack commands.
fn dispatch_stack_action(command: StackCommand) -> anyhow::Result<()> {
    match command {
        StackCommand::Release {
            crate_name,
            all,
            dry_run,
            bump,
            no_verify,
            yes,
            publish,
        } => cmd_stack_release(crate_name, all, dry_run, bump, no_verify, yes, publish),
        StackCommand::Sync {
            crate_name,
            all,
            dry_run,
            align,
        } => cmd_stack_sync(crate_name, all, dry_run, align),
        StackCommand::Quality {
            component,
            strict,
            format,
            workspace,
            ..
        } => cmd_stack_quality(component, strict, format, workspace),
        StackCommand::Gate { workspace, quiet } => cmd_stack_gate(workspace, quiet),
        StackCommand::PublishStatus {
            format,
            workspace,
            clear_cache,
        } => cmd_stack_publish_status(format, workspace, clear_cache),
        StackCommand::Drift {
            format,
            fix,
            workspace,
        } => cmd_stack_drift(format, fix, workspace),
        StackCommand::Comply {
            rule,
            fix,
            dry_run,
            format,
            workspace,
            list_rules,
        } => cmd_stack_comply(rule, fix, dry_run, format, workspace, list_rules),
        // These are handled by dispatch_stack_info
        StackCommand::Check { .. }
        | StackCommand::Status { .. }
        | StackCommand::Tree { .. }
        | StackCommand::Versions { .. } => unreachable!(),
    }
}

/// Main stack command dispatcher
pub fn cmd_stack(command: StackCommand) -> anyhow::Result<()> {
    match &command {
        StackCommand::Check { .. }
        | StackCommand::Status { .. }
        | StackCommand::Tree { .. }
        | StackCommand::Versions { .. } => dispatch_stack_info(command),
        _ => dispatch_stack_action(command),
    }
}

fn cmd_stack_check(
    _project: Option<String>,
    format: StackOutputFormat,
    strict: bool,
    verify_published: bool,
    offline: bool,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::checker::{
        format_report_json, format_report_markdown, format_report_text, StackChecker,
    };
    use stack::crates_io::CratesIoClient;

    println!("{}", "🔍 PAIML Stack Health Check".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    if offline {
        println!("{}", "📴 Offline mode - using cached data".yellow());
    }
    println!();

    // Determine workspace path
    let workspace_path = workspace.unwrap_or_else(|| PathBuf::from("."));

    // Create checker - skip crates.io verification in offline mode
    let mut checker = StackChecker::from_workspace(&workspace_path)?
        .verify_published(verify_published && !offline)
        .strict(strict);

    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    // Run check
    let report = rt.block_on(async {
        let mut client = CratesIoClient::new().with_persistent_cache();
        if offline {
            client.set_offline(true);
        }
        checker.check(&mut client).await
    })?;

    // Format output
    let output = match format {
        StackOutputFormat::Text => format_report_text(&report),
        StackOutputFormat::Json => format_report_json(&report)?,
        StackOutputFormat::Markdown => format_report_markdown(&report),
    };

    println!("{}", output);

    // Exit with error if not healthy and strict mode
    if strict && !report.is_healthy() {
        std::process::exit(1);
    }

    Ok(())
}

/// Run quality gate check before release, returns error if blocked
fn release_quality_gate() -> anyhow::Result<()> {
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker, StackQualityReport};

    println!("{}", "🔒 Running quality gate check...".dimmed());

    let workspace_path = std::env::current_dir()?
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let rt = tokio::runtime::Runtime::new()?;
    let mut components = Vec::new();

    for (_layer_name, layer_components) in LAYER_DEFINITIONS.iter() {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                if let Ok(quality) = rt.block_on(async { checker.check_component(comp_name).await })
                {
                    components.push(quality);
                }
            }
        }
    }

    let report = StackQualityReport::from_components(components);

    if !report.release_ready {
        println!();
        println!(
            "{}",
            "❌ RELEASE BLOCKED - Quality gate failed"
                .bright_red()
                .bold()
        );
        println!();
        println!(
            "The following {} component(s) are below A- threshold (SQI < 85):",
            report.blocked_components.len()
        );
        for comp in &report.blocked_components {
            println!("  • {}", comp.bright_yellow());
        }
        println!();
        println!(
            "Fix quality issues before releasing, or use --no-verify to skip (not recommended)."
        );
        anyhow::bail!(
            "Release blocked: {} component(s) below A- threshold",
            report.blocked_components.len()
        );
    }

    println!("{}", "✅ Quality gate passed".bright_green());
    println!();
    Ok(())
}

fn cmd_stack_release(
    crate_name: Option<String>,
    all: bool,
    dry_run: bool,
    bump: Option<BumpType>,
    no_verify: bool,
    _yes: bool,
    publish: bool,
) -> anyhow::Result<()> {
    use stack::checker::StackChecker;
    use stack::releaser::{format_plan_text, ReleaseConfig, ReleaseOrchestrator};

    println!("{}", "📦 PAIML Stack Release".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Early exit if no target specified
    if !all && crate_name.is_none() {
        println!("{}", "❌ Specify a crate name or use --all".red());
        return Ok(());
    }

    if !no_verify {
        release_quality_gate()?;
    } else {
        println!(
            "{}",
            "⚠️  SKIPPING quality gate check (--no-verify)".yellow()
        );
        println!();
    }

    if dry_run {
        println!(
            "{}",
            "⚠️  DRY RUN - No changes will be made".yellow().bold()
        );
        println!();
    }

    let workspace_path = PathBuf::from(".");

    // Convert CLI BumpType to releaser BumpType
    let bump_type = bump.map(|b| match b {
        BumpType::Patch => stack::releaser::BumpType::Patch,
        BumpType::Minor => stack::releaser::BumpType::Minor,
        BumpType::Major => stack::releaser::BumpType::Major,
    });

    let config = ReleaseConfig {
        bump_type,
        no_verify,
        dry_run,
        publish,
        ..Default::default()
    };

    let checker = StackChecker::from_workspace(&workspace_path)?;
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    // Plan release (we already validated crate_name or all is set)
    let plan = if all {
        orchestrator.plan_all_releases()?
    } else {
        // Safe: validated at function start that !all implies crate_name.is_some()
        let name = crate_name.expect("crate_name validated at function start");
        orchestrator.plan_release(&name)?
    };

    // Display plan
    let plan_text = format_plan_text(&plan);
    println!("{}", plan_text);

    if dry_run {
        println!();
        println!(
            "{}",
            "Dry run complete. Use without --dry-run to execute.".dimmed()
        );
        return Ok(());
    }

    // Release execution requires user confirmation and cargo publish
    // See roadmap item STACK-RELEASE for implementation plan
    println!();
    println!("{}", "Release execution not yet implemented.".yellow());
    println!("Use {} to preview the release plan.", "--dry-run".cyan());

    Ok(())
}

fn cmd_stack_status(simple: bool, format: StackOutputFormat, tree: bool) -> anyhow::Result<()> {
    use stack::checker::{
        format_report_json, format_report_markdown, format_report_text, StackChecker,
    };
    use stack::crates_io::CratesIoClient;

    println!("{}", "📊 PAIML Stack Status".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let workspace_path = PathBuf::from(".");

    // Create checker
    let mut checker = StackChecker::from_workspace(&workspace_path)?.verify_published(true);

    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    // Run check
    let report = rt.block_on(async {
        let mut client = CratesIoClient::new();
        checker.check(&mut client).await
    })?;

    if tree {
        // Display dependency tree
        println!("{}", "Dependency Tree:".bright_yellow().bold());
        if let Ok(order) = checker.topological_order() {
            for (i, name) in order.iter().enumerate() {
                println!("  {}. {}", i + 1, name.cyan());
            }
        }
        println!();
    }

    // Format output
    let output = match format {
        StackOutputFormat::Text => format_report_text(&report),
        StackOutputFormat::Json => format_report_json(&report)?,
        StackOutputFormat::Markdown => format_report_markdown(&report),
    };

    // Launch TUI only if: text format, not simple mode, and stdout is a TTY
    let is_tty = std::io::stdout().is_terminal();

    if simple || !matches!(format, StackOutputFormat::Text) || !is_tty {
        println!("{}", output);
    } else {
        // Launch interactive TUI dashboard
        stack::tui::run_dashboard(report)?;
    }

    Ok(())
}

fn cmd_stack_sync(
    crate_name: Option<String>,
    all: bool,
    dry_run: bool,
    align: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "🔄 PAIML Stack Sync".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    if dry_run {
        println!(
            "{}",
            "⚠️  DRY RUN - No changes will be made".yellow().bold()
        );
        println!();
    }

    if let Some(alignment) = &align {
        println!("Aligning dependency: {}", alignment.cyan());
    }

    if all {
        println!("Syncing all crates...");
    } else if let Some(name) = &crate_name {
        println!("Syncing crate: {}", name.cyan());
    } else {
        println!("{}", "❌ Specify a crate name or use --all".red());
        return Ok(());
    }

    // Sync logic converts path deps to crates.io versions
    // See roadmap item STACK-SYNC for implementation plan
    println!();
    println!("{}", "Sync not yet implemented.".yellow());
    println!("This will automatically convert path dependencies to crates.io versions.");

    Ok(())
}

fn cmd_stack_tree(format: &str, health: bool, filter: Option<&str>) -> anyhow::Result<()> {
    use stack::tree::{build_tree, format_ascii, format_dot, format_json, OutputFormat};

    // Parse output format
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    // Build tree
    let mut tree = build_tree();

    // Apply filter if specified
    if let Some(layer_filter) = filter {
        tree.layers.retain(|l| l.name == layer_filter);
        tree.total_crates = tree.layers.iter().map(|l| l.components.len()).sum();
    }

    // Format and print output
    let output = match output_format {
        OutputFormat::Ascii => format_ascii(&tree, health),
        OutputFormat::Json => format_json(&tree)?,
        OutputFormat::Dot => format_dot(&tree),
    };

    println!("{}", output);
    Ok(())
}

fn cmd_stack_quality(
    component: Option<String>,
    strict: bool,
    format: StackOutputFormat,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::{
        format_quality_report_json, format_quality_report_text, QualityChecker, StackQualityReport,
    };

    // Workspace is the parent directory containing all stack crates
    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    });

    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    let report = match component {
        Some(comp_name) => {
            let checker = QualityChecker::new(workspace_path.clone());
            let quality = rt.block_on(async { checker.check_component(&comp_name).await })?;
            StackQualityReport::from_components(vec![quality])
        }
        None => {
            let components = check_all_stack_components(&rt, &workspace_path);
            StackQualityReport::from_components(components)
        }
    };

    // Format output
    let output = match format {
        StackOutputFormat::Json => format_quality_report_json(&report)?,
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            format_quality_report_text(&report)
        }
    };

    println!("{}", output);

    // Exit with error if strict mode and not all A+
    if strict && !report.is_all_a_plus() {
        anyhow::bail!("Strict mode: not all components meet A+ quality standard");
    }

    // Exit with error if any component blocks release
    if !report.release_ready {
        anyhow::bail!(
            "Quality gate failed: {} component(s) below minimum threshold",
            report.blocked_components.len()
        );
    }

    Ok(())
}

/// Check all stack components from LAYER_DEFINITIONS.
fn check_all_stack_components(
    rt: &tokio::runtime::Runtime,
    workspace_path: &std::path::Path,
) -> Vec<stack::ComponentQuality> {
    use stack::tree::LAYER_DEFINITIONS;
    use stack::QualityChecker;

    let mut components = Vec::new();
    for (_layer_name, layer_components) in LAYER_DEFINITIONS {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                match rt.block_on(async { checker.check_component(comp_name).await }) {
                    Ok(quality) => components.push(quality),
                    Err(e) => {
                        eprintln!("Warning: Failed to check {}: {}", comp_name, e);
                    }
                }
            }
        }
    }
    components
}

/// Quality gate enforcement for CI/pre-commit hooks
///
/// Checks all downstream PAIML stack components and fails if any are below A- threshold.
/// This is designed to be used in pre-commit hooks or CI pipelines to prevent
/// commits/deployments when quality standards are not met.
/// Collect quality data for all stack components in the workspace.
fn collect_gate_components(
    workspace_path: &std::path::Path,
    rt: &tokio::runtime::Runtime,
    quiet: bool,
) -> Vec<stack::ComponentQuality> {
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker};

    let mut components = Vec::new();
    for (_layer_name, layer_components) in LAYER_DEFINITIONS.iter() {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                match rt.block_on(async { checker.check_component(comp_name).await }) {
                    Ok(quality) => components.push(quality),
                    Err(e) => {
                        if !quiet {
                            eprintln!("Warning: Failed to check {}: {}", comp_name, e);
                        }
                    }
                }
            }
        }
    }
    components
}

fn cmd_stack_gate(workspace: Option<PathBuf>, quiet: bool) -> anyhow::Result<()> {
    use stack::StackQualityReport;

    // Default workspace is parent of current directory (assumes we're in batuta/)
    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    });

    if !quiet {
        eprintln!("{}", "🔒 Stack Quality Gate Check".bright_cyan().bold());
        eprintln!("{}", "═".repeat(60).dimmed());
    }

    let rt = tokio::runtime::Runtime::new()?;
    let components = collect_gate_components(&workspace_path, &rt, quiet);
    let report = StackQualityReport::from_components(components);

    // Check if any components are blocked
    if !report.release_ready {
        eprintln!();
        eprintln!("{}", "❌ QUALITY GATE FAILED".bright_red().bold());
        eprintln!();
        eprintln!(
            "The following {} component(s) are below A- threshold (SQI < 85):",
            report.blocked_components.len()
        );
        for comp in &report.blocked_components {
            eprintln!("  • {}", comp.bright_yellow());
        }
        eprintln!();
        eprintln!("Run 'batuta stack quality' for detailed breakdown.");
        eprintln!("Fix quality issues before committing or deploying.");
        eprintln!();

        anyhow::bail!(
            "Quality gate failed: {} blocked component(s)",
            report.blocked_components.len()
        );
    }

    if !quiet {
        eprintln!(
            "{}",
            "✅ All components meet A- quality threshold".bright_green()
        );
        eprintln!("   Stack Quality Index: {:.1}%", report.stack_quality_index);
    }

    Ok(())
}

/// Check for newer versions of PAIML stack crates on crates.io
#[derive(Debug, serde::Serialize)]
struct CrateVersionInfo {
    name: String,
    latest: String,
    description: Option<String>,
    updated: String,
    downloads: u64,
}

#[derive(Debug, serde::Serialize)]
struct VersionReport {
    crates: Vec<CrateVersionInfo>,
    total_checked: usize,
    total_found: usize,
    timestamp: String,
}

fn cmd_stack_versions(
    outdated_only: bool,
    format: StackOutputFormat,
    offline: bool,
    _include_prerelease: bool,
) -> anyhow::Result<()> {
    use stack::crates_io::CratesIoClient;
    use stack::PAIML_CRATES;

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "📦 PAIML Stack Versions".bright_cyan().bold());
        println!("{}", "═".repeat(60).dimmed());
        if offline {
            println!("{}", "📴 Offline mode - using cached data".yellow());
        }
        println!();
        println!("{}", "Fetching latest versions from crates.io...".dimmed());
        println!();
    }

    let rt = tokio::runtime::Runtime::new()?;

    let mut client = CratesIoClient::new().with_persistent_cache();
    if offline {
        client.set_offline(true);
    }

    let mut crate_infos: Vec<CrateVersionInfo> = Vec::new();
    let mut found_count = 0;

    rt.block_on(async {
        for crate_name in PAIML_CRATES {
            match client.get_crate(crate_name).await {
                Ok(response) => {
                    found_count += 1;
                    crate_infos.push(CrateVersionInfo {
                        name: crate_name.to_string(),
                        latest: response.krate.max_version.clone(),
                        description: response.krate.description.clone(),
                        updated: response.krate.updated_at.clone(),
                        downloads: response.krate.downloads,
                    });
                }
                Err(_) => {
                    // Crate not published yet - skip silently unless verbose
                    if !outdated_only && !matches!(format, StackOutputFormat::Json) {
                        // Skip unpublished crates in normal output
                    }
                }
            }
        }
    });

    let report = VersionReport {
        crates: crate_infos,
        total_checked: PAIML_CRATES.len(),
        total_found: found_count,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    match format {
        StackOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            display_version_report_text(&report, outdated_only);
        }
    }

    Ok(())
}

fn truncate_description(desc: Option<&String>) -> String {
    desc.map(|d| {
        if d.len() > 35 {
            format!("{}...", &d[..32])
        } else {
            d.clone()
        }
    })
    .unwrap_or_else(|| "-".to_string())
}

fn display_version_report_text(report: &VersionReport, outdated_only: bool) {
    use stack::PAIML_CRATES;

    println!(
        "{:<20} {:>12} {:>12} {}",
        "Crate".bright_yellow().bold(),
        "Latest".bright_yellow().bold(),
        "Downloads".bright_yellow().bold(),
        "Description".bright_yellow().bold()
    );
    println!("{}", "─".repeat(80).dimmed());

    for info in &report.crates {
        println!(
            "{:<20} {:>12} {:>12} {}",
            info.name.cyan(),
            info.latest.green(),
            format_downloads(info.downloads),
            truncate_description(info.description.as_ref()).dimmed()
        );
    }

    println!("{}", "─".repeat(80).dimmed());
    println!();
    println!(
        "📊 Found {} of {} PAIML crates on crates.io",
        report.total_found.to_string().green(),
        report.total_checked
    );

    let unpublished: Vec<_> = PAIML_CRATES
        .iter()
        .filter(|name| !report.crates.iter().any(|c| c.name == **name))
        .collect();

    if !unpublished.is_empty() && !outdated_only {
        println!();
        println!("{}", "📝 Not yet published:".dimmed());
        for name in unpublished {
            println!("   {} {}", "•".dimmed(), name.dimmed());
        }
    }

    println!();
    println!(
        "💡 Tip: Use {} to update dependencies",
        "cargo update".cyan()
    );
}

/// Format download count for display (e.g., 1.2M, 45K)
fn format_downloads(downloads: u64) -> String {
    if downloads >= 1_000_000 {
        format!("{:.1}M", downloads as f64 / 1_000_000.0)
    } else if downloads >= 1_000 {
        format!("{:.1}K", downloads as f64 / 1_000.0)
    } else {
        downloads.to_string()
    }
}

/// Check publish status of PAIML stack repos with O(1) caching
fn cmd_stack_publish_status(
    format: StackOutputFormat,
    workspace: Option<PathBuf>,
    clear_cache: bool,
) -> anyhow::Result<()> {
    use stack::publish_status::{format_report_json, PublishStatusCache, PublishStatusScanner};

    // Workspace is parent directory (where all crates live)
    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".."))
    });

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "📦 PAIML Stack Publish Status".bright_cyan().bold());
        println!("{}", "═".repeat(65).dimmed());
        if clear_cache {
            println!("{}", "🗑️  Clearing cache...".yellow());
        }
        println!();
    }

    // Clear cache if requested
    if clear_cache {
        let mut cache = PublishStatusCache::load();
        cache.clear();
        let _ = cache.save();
    }

    // Create scanner and run
    let mut scanner = PublishStatusScanner::new(workspace_path).with_crates_io();
    let report = scanner.scan_sync()?;

    // Output based on format
    match format {
        StackOutputFormat::Json => {
            println!("{}", format_report_json(&report)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            // Colorized output
            println!(
                "{:<20} {:>10} {:>10} {:>10} {:>12}",
                "Crate".bright_yellow().bold(),
                "Local".bright_yellow().bold(),
                "crates.io".bright_yellow().bold(),
                "Git".bright_yellow().bold(),
                "Action".bright_yellow().bold()
            );
            println!("{}", "─".repeat(65).dimmed());

            for status in &report.crates {
                let local = status.local_version.as_deref().unwrap_or("-");
                let remote = status.crates_io_version.as_deref().unwrap_or("-");
                let git = status.git_status.summary();

                let action_colored = match status.action {
                    stack::PublishAction::UpToDate => "✓ up to date".green(),
                    stack::PublishAction::NeedsPublish => "📦 PUBLISH".bright_red().bold(),
                    stack::PublishAction::NeedsCommit => "📝 commit".yellow(),
                    stack::PublishAction::LocalBehind => "⚠️  behind".yellow(),
                    stack::PublishAction::NotPublished => "🆕 new".cyan(),
                    stack::PublishAction::Error => "❌ error".red(),
                };

                println!(
                    "{:<20} {:>10} {:>10} {:>10} {}",
                    status.name.cyan(),
                    local,
                    remote,
                    git.dimmed(),
                    action_colored
                );
            }

            println!("{}", "─".repeat(65).dimmed());
            println!();

            // Summary
            println!(
                "📊 {} crates: {} {}, {} {}, {} up-to-date",
                report.total,
                report.needs_publish.to_string().bright_red().bold(),
                "publish".red(),
                report.needs_commit.to_string().yellow(),
                "commit".yellow(),
                report.up_to_date.to_string().green()
            );
            println!(
                "⚡ {}ms (cache: {} hits, {} misses)",
                report.elapsed_ms,
                report.cache_hits.to_string().green(),
                report.cache_misses
            );
        }
    }

    Ok(())
}

fn cmd_stack_drift(
    format: StackOutputFormat,
    fix: bool,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::crates_io::CratesIoClient;
    use stack::drift::{format_drift_json, DriftChecker};

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "🔍 PAIML Stack Drift Detection".bright_cyan().bold());
        println!("{}", "═".repeat(70).dimmed());
        println!();
    }

    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    let drifts = rt.block_on(async {
        let mut client = CratesIoClient::new().with_persistent_cache();
        let mut checker = DriftChecker::new();
        checker.detect_drift(&mut client).await
    })?;

    if drifts.is_empty() {
        match format {
            StackOutputFormat::Json => {
                println!("{}", format_drift_json(&[])?);
            }
            _ => {
                println!(
                    "{}",
                    "✅ No drift detected - all stack crates are using latest versions!"
                        .green()
                        .bold()
                );
            }
        }
        return Ok(());
    }

    // Output based on format
    match format {
        StackOutputFormat::Json => {
            println!("{}", format_drift_json(&drifts)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            display_drift_table(&drifts);
            if fix {
                display_drift_fix_commands(&drifts, workspace);
            } else {
                println!();
                println!("{}", "Run with --fix to generate fix commands.".dimmed());
            }
        }
    }

    Ok(())
}

/// Display the drift report table with colored output.
fn display_drift_table(drifts: &[stack::DriftReport]) {
    println!(
        "{:<20} {:>10} {:<15} {:>12} {:>12} {:>8}",
        "Crate".bright_yellow().bold(),
        "Version".bright_yellow().bold(),
        "Dependency".bright_yellow().bold(),
        "Uses".bright_yellow().bold(),
        "Latest".bright_yellow().bold(),
        "Severity".bright_yellow().bold()
    );
    println!("{}", "─".repeat(70).dimmed());

    for drift in drifts {
        let severity_colored = match drift.severity {
            stack::DriftSeverity::Major => "MAJOR".bright_red().bold(),
            stack::DriftSeverity::Minor => "MINOR".yellow(),
            stack::DriftSeverity::Patch => "PATCH".dimmed(),
        };
        println!(
            "{:<20} {:>10} {:<15} {:>12} {:>12} {}",
            drift.crate_name.cyan(),
            drift.crate_version.dimmed(),
            drift.dependency.bright_white(),
            drift.uses_version.red(),
            drift.latest_version.green(),
            severity_colored
        );
    }

    println!("{}", "─".repeat(70).dimmed());
    println!();

    let major_count = drifts
        .iter()
        .filter(|d| matches!(d.severity, stack::DriftSeverity::Major))
        .count();
    let minor_count = drifts
        .iter()
        .filter(|d| matches!(d.severity, stack::DriftSeverity::Minor))
        .count();

    println!(
        "📊 {} drift issues: {} {}, {} {}",
        drifts.len(),
        major_count.to_string().bright_red().bold(),
        "major".red(),
        minor_count.to_string().yellow(),
        "minor".yellow()
    );
}

/// Display fix commands for drift issues grouped by crate.
fn display_drift_fix_commands(drifts: &[stack::DriftReport], workspace: Option<PathBuf>) {
    println!();
    println!(
        "{}",
        "🔧 Fix Commands (run in each crate directory):".bright_cyan()
    );
    println!("{}", "─".repeat(70).dimmed());

    let ws = workspace.unwrap_or_else(|| PathBuf::from(".."));

    let mut by_crate: std::collections::HashMap<&str, Vec<&stack::DriftReport>> =
        std::collections::HashMap::new();
    for drift in drifts {
        by_crate.entry(&drift.crate_name).or_default().push(drift);
    }

    for (crate_name, crate_drifts) in by_crate {
        let crate_path = ws.join(crate_name);
        println!();
        println!("# {} ({})", crate_name.cyan().bold(), crate_path.display());
        for drift in crate_drifts {
            let dep = &drift.dependency;
            let old_minor = extract_minor_version(&drift.uses_version);
            let new_minor = extract_minor_version(&drift.latest_version);
            println!(
                "sed -i 's/{} = \"\\([^0-9]*\\){}\\([^\"]*\\)\"/{} = \"\\1{}\\2\"/g' {}/Cargo.toml",
                dep,
                old_minor,
                dep,
                new_minor,
                crate_path.display()
            );
        }
    }

    println!();
    println!(
        "{}",
        "After fixing, run 'cargo update' in each crate directory.".dimmed()
    );
}

/// Extract minor version (e.g., "0.10" from "0.10.1" or "^0.10")
fn extract_minor_version(version: &str) -> String {
    let cleaned = version
        .trim_start_matches('^')
        .trim_start_matches('~')
        .trim_start_matches('=')
        .trim_start_matches('>')
        .trim_start_matches('<')
        .trim();

    let parts: Vec<&str> = cleaned.split('.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        cleaned.to_string()
    }
}

fn cmd_stack_comply(
    rule: Option<String>,
    fix: bool,
    dry_run: bool,
    format: ComplyOutputFormat,
    workspace: Option<PathBuf>,
    list_rules: bool,
) -> anyhow::Result<()> {
    use crate::comply::{ComplyConfig, ComplyReportFormat, StackComplyEngine};

    let workspace_path = workspace.unwrap_or_else(|| {
        // Try to find workspace root (parent of current directory)
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    });

    println!(
        "{}",
        "🔍 PAIML Stack Compliance Check".bright_cyan().bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Create engine
    let config = ComplyConfig::load_or_default(&workspace_path);
    let mut engine = StackComplyEngine::new(config);

    // List rules if requested
    if list_rules {
        println!("{}", "Available compliance rules:".bright_white().bold());
        println!();
        for (id, description) in engine.available_rules() {
            println!("  {} - {}", id.bright_yellow(), description);
        }
        return Ok(());
    }

    // Discover projects
    println!("{}", "Discovering projects...".dimmed());
    engine.discover_projects(&workspace_path)?;
    println!(
        "  Found {} projects ({} PAIML crates)",
        engine.projects().len(),
        engine
            .projects()
            .iter()
            .filter(|p| p.is_paiml_crate)
            .count()
    );
    println!();

    // Run checks or fixes
    let report = if fix || dry_run {
        if dry_run {
            println!("{}", "⚠️  DRY RUN - No changes will be made".yellow().bold());
        } else {
            println!("{}", "🔧 Attempting to fix violations...".bright_yellow());
        }
        engine.fix_all(dry_run)
    } else if let Some(rule_id) = rule {
        println!("Checking rule: {}", rule_id.bright_yellow());
        engine.check_rule(&rule_id)
    } else {
        println!("{}", "Running all compliance checks...".dimmed());
        engine.check_all()
    };

    // Convert format
    let output_format = match format {
        ComplyOutputFormat::Text => ComplyReportFormat::Text,
        ComplyOutputFormat::Json => ComplyReportFormat::Json,
        ComplyOutputFormat::Markdown => ComplyReportFormat::Markdown,
        ComplyOutputFormat::Html => ComplyReportFormat::Html,
    };

    // Output report
    println!();
    println!("{}", report.format(output_format));

    // Summary
    if report.is_compliant() {
        println!(
            "{}",
            "✅ All compliance checks passed!".bright_green().bold()
        );
    } else {
        println!(
            "{}",
            format!(
                "❌ {} violations found across {} projects",
                report.summary.total_violations, report.summary.failing_projects
            )
            .bright_red()
            .bold()
        );

        if report.summary.fixable_violations > 0 {
            println!(
                "   {} violations are auto-fixable (run with --fix)",
                report.summary.fixable_violations
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_minor_version_simple() {
        assert_eq!(extract_minor_version("0.10.1"), "0.10");
        assert_eq!(extract_minor_version("1.2.3"), "1.2");
    }

    #[test]
    fn test_extract_minor_version_with_prefix() {
        assert_eq!(extract_minor_version("^0.10"), "0.10");
        assert_eq!(extract_minor_version("~1.2"), "1.2");
        assert_eq!(extract_minor_version("=2.0.0"), "2.0");
    }

    #[test]
    fn test_format_downloads() {
        assert_eq!(format_downloads(500), "500");
        assert_eq!(format_downloads(1500), "1.5K");
        assert_eq!(format_downloads(1_500_000), "1.5M");
    }
}
