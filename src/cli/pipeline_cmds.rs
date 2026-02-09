//! Pipeline command handlers
//!
//! Extracted from main.rs for file size compliance.
//! Contains: cmd_init, cmd_analyze, cmd_transpile, cmd_optimize, cmd_validate, cmd_build, cmd_report

use crate::analyzer::analyze_project;
use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::pipeline::{PipelineContext, PipelineStage, ValidationStage};
use crate::report;
use crate::tools::{self, ToolInfo, ToolRegistry};
use crate::types::{Language, PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState};
use std::path::{Path, PathBuf};
use tracing::warn;

/// CLI report format
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ReportFormat {
    /// HTML report with charts
    #[default]
    Html,
    /// Markdown report
    Markdown,
    /// JSON data
    Json,
    /// Plain text report
    Text,
}

/// CLI optimization profile
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OptimizationProfile {
    /// Fast compilation, basic optimizations
    Fast,
    /// Balanced compilation and performance
    #[default]
    Balanced,
    /// Maximum performance, slower compilation
    Aggressive,
}

// ============================================================================
// Init Command
// ============================================================================

pub fn cmd_init(source: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
    println!(
        "{}",
        "🚀 Initializing Batuta project...".bright_cyan().bold()
    );
    println!();

    // Analyze the source project
    println!("{}", "Analyzing source project...".dimmed());
    let analysis = analyze_project(&source, true, true, true)?;

    println!("{} Source: {:?}", "✓".bright_green(), source);
    if let Some(lang) = &analysis.primary_language {
        println!(
            "{} Detected language: {}",
            "✓".bright_green(),
            format!("{}", lang).cyan()
        );
    }
    println!();

    // Determine output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut dir = source.clone();
        dir.push("rust-output");
        dir
    });

    // Create configuration from analysis
    let mut config = BatutaConfig::from_analysis(&analysis);

    // Set output directory
    config.transpilation.output_dir = output_dir.clone();

    // Save configuration
    let config_path = source.join("batuta.toml");
    config.save(&config_path)?;

    println!(
        "{} Created configuration: {:?}",
        "✓".bright_green(),
        config_path
    );

    // Create output directory structure
    std::fs::create_dir_all(&output_dir)?;
    std::fs::create_dir_all(output_dir.join("src"))?;

    println!(
        "{} Created output directory: {:?}",
        "✓".bright_green(),
        output_dir
    );
    println!();

    // Display configuration summary
    display_init_summary(&config, &analysis);

    Ok(())
}

fn display_init_summary(config: &BatutaConfig, analysis: &ProjectAnalysis) {
    println!("{}", "📋 Configuration Summary".bright_yellow().bold());
    println!("{}", "=".repeat(50));
    println!();
    println!("{}: {}", "Project name".bold(), config.project.name.cyan());
    println!(
        "{}: {}",
        "Primary language".bold(),
        config
            .project
            .primary_language
            .as_ref()
            .unwrap_or(&"Unknown".to_string())
            .cyan()
    );
    println!(
        "{}: {:?}",
        "Output directory".bold(),
        config.transpilation.output_dir
    );
    println!();

    // Display transpilation settings
    println!("{}", "Transpilation:".bright_yellow());
    println!(
        "  {} Incremental: {}",
        "•".bright_blue(),
        config.transpilation.incremental.to_string().cyan()
    );
    println!(
        "  {} Caching: {}",
        "•".bright_blue(),
        config.transpilation.cache.to_string().cyan()
    );

    if analysis.has_ml_dependencies() {
        println!(
            "  {} NumPy → Trueno: {}",
            "•".bright_blue(),
            "enabled".green()
        );
        println!(
            "  {} sklearn → Aprender: {}",
            "•".bright_blue(),
            "enabled".green()
        );
        println!(
            "  {} PyTorch → Realizar: {}",
            "•".bright_blue(),
            "enabled".green()
        );
    }
    println!();

    // Display optimization settings
    println!("{}", "Optimization:".bright_yellow());
    println!(
        "  {} Profile: {}",
        "•".bright_blue(),
        config.optimization.profile.cyan()
    );
    println!(
        "  {} SIMD: {}",
        "•".bright_blue(),
        config.optimization.enable_simd.to_string().cyan()
    );
    println!(
        "  {} GPU: {}",
        "•".bright_blue(),
        if config.optimization.enable_gpu {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!();

    // Next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Edit {} to customize settings",
        "1.".bright_blue(),
        "batuta.toml".cyan()
    );
    println!(
        "  {} Run {} to convert your code",
        "2.".bright_blue(),
        "batuta transpile".cyan()
    );
    println!(
        "  {} Run {} to optimize performance",
        "3.".bright_blue(),
        "batuta optimize".cyan()
    );
    println!();
}

// ============================================================================
// Analyze Command
// ============================================================================

pub fn cmd_analyze(
    path: PathBuf,
    tdg: bool,
    languages: bool,
    dependencies: bool,
) -> anyhow::Result<()> {
    println!("{}", "🔍 Analyzing project...".bright_cyan().bold());
    println!();

    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    state.start_phase(WorkflowPhase::Analysis);
    state.save(&state_file)?;

    let analysis = analyze_project(&path, tdg, languages, dependencies)?;

    // Display results based on flags
    display_analysis_results(&analysis);

    // Update and save workflow state
    state.complete_phase(WorkflowPhase::Analysis);

    // Create a default config if not exists
    let config_path = path.join("batuta.toml");
    if !config_path.exists() {
        let config = BatutaConfig::from_analysis(&analysis);
        config.save(&config_path)?;
        println!(
            "{} Created default configuration: {:?}",
            "✓".bright_green(),
            config_path
        );
        println!();
    }

    state.save(&state_file)?;

    super::workflow::display_workflow_progress(&state);
    display_analyze_next_steps();

    Ok(())
}

/// Display project analysis results
pub fn display_analysis_results(analysis: &ProjectAnalysis) {
    println!("{}", "📊 Analysis Results".bright_green().bold());
    println!("{}", "=".repeat(50));
    println!();

    // Project info
    println!("{}: {:?}", "Project path".bold(), analysis.root_path);
    println!(
        "{}: {}",
        "Total files".bold(),
        analysis.total_files.to_string().cyan()
    );
    println!(
        "{}: {}",
        "Total lines".bold(),
        analysis.total_lines.to_string().cyan()
    );
    println!();

    // Languages
    display_language_info(analysis);

    // Dependencies
    display_dependency_info(analysis);

    // TDG Score
    display_tdg_score(analysis);
}

/// Display language detection results
fn display_language_info(analysis: &ProjectAnalysis) {
    if analysis.languages.is_empty() {
        return;
    }

    println!("{}", "Languages Detected:".bright_yellow().bold());
    for lang_stat in &analysis.languages {
        println!(
            "  {} {} - {} files, {} lines ({:.1}%)",
            "•".bright_blue(),
            format!("{}", lang_stat.language).cyan(),
            lang_stat.file_count,
            lang_stat.line_count,
            lang_stat.percentage
        );
    }
    println!();

    // Show primary language recommendation
    if let Some(primary) = &analysis.primary_language {
        println!(
            "{}: {} ({})",
            "Primary language".bold(),
            format!("{}", primary).cyan(),
            "highest line count".dimmed()
        );

        // Map to transpiler
        let transpiler = match primary {
            Language::Python => Some("depyler"),
            Language::Shell => Some("bashrs"),
            Language::C | Language::Cpp => Some("decy"),
            _ => None,
        };

        if let Some(t) = transpiler {
            println!(
                "{}: {}",
                "Recommended transpiler".bold(),
                t.bright_green()
            );
        }
        println!();
    }
}

/// Display dependency detection results
fn display_dependency_info(analysis: &ProjectAnalysis) {
    if analysis.dependencies.is_empty() {
        return;
    }

    println!("{}", "Dependencies:".bright_yellow().bold());
    for dep in &analysis.dependencies {
        let count_str = if let Some(count) = dep.count {
            format!(" ({} packages)", count)
        } else {
            String::new()
        };
        println!(
            "  {} {}{}",
            "•".bright_blue(),
            format!("{}", dep.manager).cyan(),
            count_str.yellow()
        );
        println!("    {}: {:?}", "File".dimmed(), dep.file_path);
    }
    println!();

    if analysis.has_ml_dependencies() {
        println!(
            "  {} {}",
            "ℹ".bright_blue(),
            "ML frameworks detected - consider Aprender/Realizar for ML code".bright_yellow()
        );
        println!();
    }
}

/// Display TDG quality score
fn display_tdg_score(analysis: &ProjectAnalysis) {
    let Some(score) = analysis.tdg_score else {
        return;
    };

    let grade = super::calculate_tdg_grade(score);
    let grade_colored = match grade {
        super::TdgGrade::APlus => format!("{}", grade).bright_green(),
        super::TdgGrade::A => format!("{}", grade).green(),
        super::TdgGrade::B | super::TdgGrade::C => format!("{}", grade).yellow(),
        super::TdgGrade::D => format!("{}", grade).red(),
    };

    println!("{}", "Quality Score:".bright_yellow().bold());
    println!(
        "  {} TDG Score: {}/100 ({})",
        "•".bright_blue(),
        format!("{:.1}", score).cyan(),
        grade_colored
    );
    println!();
}

fn display_analyze_next_steps() {
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Run {} to convert your code",
        "1.".bright_blue(),
        "batuta transpile".cyan()
    );
    println!(
        "  {} Run {} for detailed dependency analysis",
        "2.".bright_blue(),
        "batuta analyze --tdg".cyan()
    );
    println!();
}

// ============================================================================
// Transpile Command
// ============================================================================

pub fn cmd_transpile(
    incremental: bool,
    cache: bool,
    modules: Option<Vec<String>>,
    ruchy: bool,
    repl: bool,
) -> anyhow::Result<()> {
    println!("{}", "🔄 Transpiling code...".bright_cyan().bold());
    println!();

    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check prerequisites
    let config = match check_transpile_prerequisites(&state) {
        Ok(c) => c,
        Err(e) => {
            state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
            state.save(&state_file)?;
            return Ok(());
        }
    };

    state.start_phase(WorkflowPhase::Transpilation);
    state.save(&state_file)?;

    // Setup transpiler
    let (_tools, _primary_lang, transpiler) = match setup_transpiler(&config) {
        Ok(t) => t,
        Err(e) => {
            state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
            state.save(&state_file)?;
            return Err(e);
        }
    };

    // Display settings and create directories
    display_transpilation_settings(&config, incremental, cache, ruchy, &modules);
    std::fs::create_dir_all(&config.transpilation.output_dir)?;
    std::fs::create_dir_all(config.transpilation.output_dir.join("src"))?;

    println!("{}", "🚀 Starting transpilation...".bright_green().bold());
    println!();

    // Build and display command
    let owned_args = super::build_transpiler_args(&config, incremental, cache, ruchy, &modules);
    let args: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

    println!("{}", "Executing:".dimmed());
    println!(
        "  {} {} {}",
        "$".dimmed(),
        transpiler.name.cyan(),
        args.join(" ").dimmed()
    );
    println!();
    println!("{}", "Transpiling...".bright_yellow());

    // Run transpiler and handle result
    let result = tools::run_tool(&transpiler.name, &args, Some(&config.source.path));
    handle_transpile_result(result, &mut state, &state_file, &config, &transpiler, repl)
}

fn check_transpile_prerequisites(state: &WorkflowState) -> anyhow::Result<BatutaConfig> {
    // Check if analysis phase is completed
    if !state.is_phase_completed(WorkflowPhase::Analysis) {
        println!(
            "{}",
            "⚠️  Analysis phase not completed!".yellow().bold()
        );
        println!();
        println!(
            "Run {} first to analyze your project.",
            "batuta analyze <path>".cyan()
        );
        println!();
        super::workflow::display_workflow_progress(state);
        anyhow::bail!("Analysis phase not completed");
    }

    // Load configuration
    let config_path = PathBuf::from("batuta.toml");
    if !config_path.exists() {
        println!("{}", "⚠️  No configuration file found!".yellow().bold());
        println!();
        println!(
            "Run {} to create a configuration file.",
            "batuta init".cyan()
        );
        println!();
        anyhow::bail!("No batuta.toml configuration file");
    }

    BatutaConfig::load(&config_path)
}

fn setup_transpiler(
    config: &BatutaConfig,
) -> anyhow::Result<(ToolRegistry, Language, ToolInfo)> {
    // Detect available tools
    let tools = ToolRegistry::detect();

    // Get primary language
    let primary_lang = config
        .project
        .primary_language
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(Language::Other("unknown".to_string()));

    // Get appropriate transpiler
    let transpiler = match tools.get_transpiler_for_language(&primary_lang) {
        Some(t) => t.clone(),
        None => {
            // handle_missing_tools always returns Err, so this branch never completes
            handle_missing_tools(&tools, &primary_lang)?;
            unreachable!("handle_missing_tools always returns Err")
        }
    };

    println!(
        "{} Using transpiler: {} ({})",
        "✓".bright_green(),
        transpiler.name.cyan(),
        transpiler.version.as_ref().unwrap_or(&"unknown".to_string())
    );
    println!();

    Ok((tools, primary_lang, transpiler))
}

fn handle_missing_tools(tools: &ToolRegistry, primary_lang: &Language) -> anyhow::Result<()> {
    println!("{}", "❌ No transpiler available!".red().bold());
    println!();
    println!("{}", "Available tools:".bright_yellow());
    for tool in tools.available_tools() {
        println!("  {} {}", "•".bright_blue(), tool.cyan());
    }
    println!();
    println!(
        "{}: Install the appropriate transpiler for {}",
        "Hint".bold(),
        format!("{}", primary_lang).cyan()
    );
    println!("  {} cargo install depyler", "Python:".dimmed());
    println!("  {} cargo install bashrs", "Shell:".dimmed());
    println!("  {} cargo install decy", "C/C++:".dimmed());
    println!();
    anyhow::bail!("No transpiler available for {}", primary_lang)
}

fn display_transpilation_settings(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) {
    println!("{}", "Transpilation Settings:".bright_yellow().bold());
    println!(
        "  {} Source: {:?}",
        "•".bright_blue(),
        config.source.path
    );
    println!(
        "  {} Output: {:?}",
        "•".bright_blue(),
        config.transpilation.output_dir
    );
    println!(
        "  {} Incremental: {}",
        "•".bright_blue(),
        if incremental || config.transpilation.incremental {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} Caching: {}",
        "•".bright_blue(),
        if cache || config.transpilation.cache {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );

    if let Some(mods) = modules {
        println!(
            "  {} Modules: {}",
            "•".bright_blue(),
            mods.join(", ").cyan()
        );
    }

    if ruchy || config.transpilation.use_ruchy {
        println!("  {} Target: {}", "•".bright_blue(), "Ruchy".cyan());
        if let Some(strictness) = &config.transpilation.ruchy_strictness {
            println!("  {} Strictness: {}", "•".bright_blue(), strictness.cyan());
        }
    }
    println!();
}

fn handle_transpile_result(
    result: Result<String, anyhow::Error>,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &ToolInfo,
    repl: bool,
) -> anyhow::Result<()> {
    match result {
        Ok(output) => handle_transpile_success(output, state, state_file, config, repl),
        Err(e) => handle_transpile_failure(e, state, state_file, config, transpiler),
    }
}

fn handle_transpile_success(
    output: String,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    repl: bool,
) -> anyhow::Result<()> {
    println!();
    println!(
        "{}",
        "✅ Transpilation completed successfully!"
            .bright_green()
            .bold()
    );
    println!();

    // Display transpiler output
    if !output.trim().is_empty() {
        println!("{}", "Transpiler output:".bright_yellow());
        println!("{}", "─".repeat(50).dimmed());
        for line in output.lines().take(20) {
            println!("  {}", line.dimmed());
        }
        if output.lines().count() > 20 {
            println!(
                "  {} ... ({} more lines)",
                "...".dimmed(),
                output.lines().count() - 20
            );
        }
        println!("{}", "─".repeat(50).dimmed());
        println!();
    }

    // Complete transpilation phase
    state.complete_phase(WorkflowPhase::Transpilation);
    state.save(state_file)?;

    super::workflow::display_workflow_progress(state);

    // Show next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Check output directory: {:?}",
        "1.".bright_blue(),
        config.transpilation.output_dir
    );
    println!(
        "  {} Run {} to optimize",
        "2.".bright_blue(),
        "batuta optimize".cyan()
    );
    println!(
        "  {} Run {} to validate",
        "3.".bright_blue(),
        "batuta validate".cyan()
    );
    println!();

    // Start REPL if requested
    if repl {
        println!("{}", "🔬 Starting Ruchy REPL...".bright_cyan().bold());
        warn!("Ruchy REPL not yet implemented");
    }

    Ok(())
}

fn handle_transpile_failure(
    e: anyhow::Error,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &ToolInfo,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "❌ Transpilation failed!".red().bold());
    println!();
    println!("{}: {}", "Error".bold(), e.to_string().red());
    println!();

    state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
    state.save(state_file)?;

    super::workflow::display_workflow_progress(state);

    // Provide helpful troubleshooting
    println!("{}", "💡 Troubleshooting:".bright_yellow().bold());
    println!(
        "  {} Verify {} is properly installed",
        "•".bright_blue(),
        transpiler.name.cyan()
    );
    println!(
        "  {} Check that source path is correct: {:?}",
        "•".bright_blue(),
        config.source.path
    );
    println!(
        "  {} Try running with {} for more details",
        "•".bright_blue(),
        "--verbose".cyan()
    );
    println!(
        "  {} See transpiler docs: {}",
        "•".bright_blue(),
        format!("https://github.com/paiml/{}", transpiler.name).cyan()
    );
    println!();

    Err(e)
}

// ============================================================================
// Optimize Command
// ============================================================================

pub fn cmd_optimize(
    enable_gpu: bool,
    enable_simd: bool,
    profile: OptimizationProfile,
    gpu_threshold: usize,
) -> anyhow::Result<()> {
    println!("{}", "⚡ Optimizing code...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if transpilation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Transpilation) {
        println!(
            "{}",
            "⚠️  Transpilation phase not completed!".yellow().bold()
        );
        println!();
        println!(
            "Run {} first to transpile your project.",
            "batuta transpile".cyan()
        );
        println!();
        super::workflow::display_workflow_progress(&state);
        return Ok(());
    }

    // Start optimization phase
    state.start_phase(WorkflowPhase::Optimization);
    state.save(&state_file)?;

    // Display optimization settings
    println!("{}", "Optimization Settings:".bright_yellow().bold());
    println!("  {} Profile: {:?}", "•".bright_blue(), profile);
    println!(
        "  {} SIMD vectorization: {}",
        "•".bright_blue(),
        if enable_simd {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} GPU acceleration: {}",
        "•".bright_blue(),
        if enable_gpu {
            format!("enabled (threshold: {})", gpu_threshold).green()
        } else {
            "disabled".to_string().dimmed()
        }
    );
    println!();

    // BATUTA-007: Optimization engine planned for Phase 3
    warn!("Optimization execution not yet implemented - Phase 3 (BATUTA-007)");
    println!(
        "{}",
        "🚧 Optimization engine coming soon!".bright_yellow().bold()
    );
    println!();
    println!("{}", "Planned optimizations:".dimmed());
    println!("  {} SIMD vectorization via Trueno", "•".dimmed());
    println!("  {} GPU dispatch for large operations", "•".dimmed());
    println!("  {} Memory layout optimization", "•".dimmed());
    println!("  {} MoE backend selection", "•".dimmed());
    println!();

    // For now, mark as completed (once implemented, this will be conditional on success)
    state.complete_phase(WorkflowPhase::Optimization);
    state.save(&state_file)?;

    // Display workflow progress
    super::workflow::display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Run {} to verify equivalence",
        "1.".bright_blue(),
        "batuta validate".cyan()
    );
    println!(
        "  {} Run {} to build final binary",
        "2.".bright_blue(),
        "batuta build --release".cyan()
    );
    println!();

    Ok(())
}

// ============================================================================
// Validate Command
// ============================================================================

pub fn cmd_validate(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) -> anyhow::Result<()> {
    println!("{}", "✅ Validating equivalence...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if optimization phase is completed
    if !state.is_phase_completed(WorkflowPhase::Optimization) {
        println!(
            "{}",
            "⚠️  Optimization phase not completed!".yellow().bold()
        );
        println!();
        println!(
            "Run {} first to optimize your project.",
            "batuta optimize".cyan()
        );
        println!();
        super::workflow::display_workflow_progress(&state);
        return Ok(());
    }

    // Start validation phase
    state.start_phase(WorkflowPhase::Validation);
    state.save(&state_file)?;

    // Display validation settings
    display_validation_settings(trace_syscalls, diff_output, run_original_tests, benchmark);

    // Implement validation with Renacer (BATUTA-011)
    let mut validation_passed = true;

    if trace_syscalls && !run_syscall_tracing(run_original_tests) {
        validation_passed = false;
    }

    if diff_output {
        println!("{}", "📊 Output comparison:".dimmed());
        println!("  {} Not yet implemented", "•".dimmed());
        println!();
    }

    if run_original_tests {
        println!("{}", "🧪 Running original test suite:".dimmed());
        println!("  {} Not yet implemented", "•".dimmed());
        println!();
    }

    if benchmark {
        println!("{}", "⚡ Performance benchmarking:".dimmed());
        println!("  {} Not yet implemented", "•".dimmed());
        println!();
    }

    // Mark as completed only if validation passed
    if validation_passed {
        state.complete_phase(WorkflowPhase::Validation);
    } else {
        state.fail_phase(
            WorkflowPhase::Validation,
            "Validation checks failed".to_string(),
        );
    }
    state.save(&state_file)?;

    // Display workflow progress
    super::workflow::display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Run {} to build final binary",
        "1.".bright_blue(),
        "batuta build --release".cyan()
    );
    println!(
        "  {} Run {} to generate report",
        "2.".bright_blue(),
        "batuta report".cyan()
    );
    println!();

    Ok(())
}

/// Run Renacer syscall tracing validation. Returns true if passed.
fn run_syscall_tracing(run_original_tests: bool) -> bool {
    println!("{}", "🔍 Running Renacer syscall tracing...".bright_cyan());

    let original_binary = std::path::Path::new("./original_binary");
    let transpiled_binary = std::path::Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for comparison".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    println!("  {} Tracing original binary...", "•".bright_blue());
    println!("  {} Tracing transpiled binary...", "•".bright_blue());
    println!("  {} Comparing syscall traces...", "•".bright_blue());

    let ctx = PipelineContext::new(PathBuf::from("."), PathBuf::from("."));
    let stage = ValidationStage::new(true, run_original_tests);

    match tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime")
        .block_on(stage.execute(ctx))
    {
        Ok(result_ctx) => {
            if let Some(eq) = result_ctx.metadata.get("syscall_equivalence") {
                if eq.as_bool() == Some(true) {
                    println!(
                        "{}",
                        "  ✅ Syscall traces match - semantic equivalence verified".green()
                    );
                    println!();
                    true
                } else {
                    println!(
                        "{}",
                        "  ❌ Syscall traces differ - equivalence NOT verified".red()
                    );
                    println!();
                    false
                }
            } else {
                println!(
                    "{}",
                    "  ⚠️  Syscall tracing skipped (binaries not found)".yellow()
                );
                println!();
                true
            }
        }
        Err(e) => {
            println!("{}", format!("  ❌ Validation error: {}", e).red());
            println!();
            false
        }
    }
}

/// Display validation settings as a formatted list.
fn display_validation_settings(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) {
    let settings = [
        ("Syscall tracing", trace_syscalls),
        ("Diff output", diff_output),
        ("Original tests", run_original_tests),
        ("Benchmarks", benchmark),
    ];
    println!("{}", "Validation Settings:".bright_yellow().bold());
    for (label, enabled) in settings {
        println!(
            "  {} {}: {}",
            "•".bright_blue(),
            label,
            if enabled {
                "enabled".green()
            } else {
                "disabled".dimmed()
            }
        );
    }
    println!();
}

// ============================================================================
// Build Command
// ============================================================================

fn run_cargo_build(
    project_dir: &Path,
    release: bool,
    target: Option<&str>,
    wasm: bool,
    extra_flags: &[String],
) -> anyhow::Result<()> {
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").current_dir(project_dir);

    if wasm {
        cmd.arg("--target").arg("wasm32-unknown-unknown");
    } else if let Some(t) = target {
        cmd.arg("--target").arg(t);
    }
    if release {
        cmd.arg("--release");
    }
    for flag in extra_flags {
        cmd.arg(flag);
    }

    // Display the command being run
    let mut display = String::from("cargo build");
    if release {
        display.push_str(" --release");
    }
    if wasm {
        display.push_str(" --target wasm32-unknown-unknown");
    } else if let Some(t) = target {
        display.push_str(&format!(" --target {}", t));
    }
    for flag in extra_flags {
        display.push(' ');
        display.push_str(flag);
    }
    println!("{} {}", "Running:".bright_yellow(), display.cyan());
    println!();

    let status = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to execute cargo (is it in PATH?): {}", e))?;

    if status.success() {
        println!();
        println!(
            "{}",
            "✅ Build completed successfully!".bright_green().bold()
        );
        Ok(())
    } else {
        let code = status
            .code()
            .map_or("signal".to_string(), |c| c.to_string());
        println!();
        println!("{} Build failed with exit code: {}", "✗".red(), code);
        anyhow::bail!("cargo build failed (exit {})", code)
    }
}

pub fn cmd_build(release: bool, target: Option<String>, wasm: bool) -> anyhow::Result<()> {
    println!("{}", "🔨 Building Rust project...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if validation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Validation) {
        println!("{}", "⚠️  Validation phase not completed!".yellow().bold());
        println!();
        println!(
            "Run {} first to validate your project.",
            "batuta validate".cyan()
        );
        println!();
        super::workflow::display_workflow_progress(&state);
        return Ok(());
    }

    // Start deployment phase
    state.start_phase(WorkflowPhase::Deployment);
    state.save(&state_file)?;

    // Display build settings
    println!("{}", "Build Settings:".bright_yellow().bold());
    println!(
        "  {} Build mode: {}",
        "•".bright_blue(),
        if release {
            "release".green()
        } else {
            "debug".dimmed()
        }
    );
    if let Some(t) = &target {
        println!("  {} Target: {}", "•".bright_blue(), t.cyan());
    }
    println!(
        "  {} WebAssembly: {}",
        "•".bright_blue(),
        if wasm {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!();

    // Load project config to find the transpiled output directory
    let config_path = PathBuf::from("batuta.toml");
    let config = if config_path.exists() {
        BatutaConfig::load(&config_path)?
    } else {
        BatutaConfig::default()
    };

    let output_dir = &config.transpilation.output_dir;
    if !output_dir.join("Cargo.toml").exists() {
        println!(
            "{} No Cargo.toml found in {}",
            "✗".red(),
            output_dir.display()
        );
        println!();
        println!(
            "Run {} first to generate the Rust project.",
            "batuta transpile".cyan()
        );
        state.fail_phase(
            WorkflowPhase::Deployment,
            format!("No Cargo.toml in {}", output_dir.display()),
        );
        state.save(&state_file)?;
        anyhow::bail!(
            "No Cargo.toml in transpiled output directory: {}",
            output_dir.display()
        );
    }

    println!(
        "  {} Project: {}",
        "•".bright_blue(),
        output_dir.display()
    );
    println!();

    // Execute cargo build in the transpiled project
    match run_cargo_build(output_dir, release, target.as_deref(), wasm, &config.build.cargo_flags) {
        Ok(()) => {
            state.complete_phase(WorkflowPhase::Deployment);
            state.save(&state_file)?;
        }
        Err(e) => {
            state.fail_phase(WorkflowPhase::Deployment, e.to_string());
            state.save(&state_file)?;
            return Err(e);
        }
    }

    // Display workflow progress
    super::workflow::display_workflow_progress(&state);

    println!("{}", "🎉 Migration Complete!".bright_green().bold());
    println!();
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!(
        "  {} Run {} to generate migration report",
        "1.".bright_blue(),
        "batuta report".cyan()
    );
    println!(
        "  {} Check your output directory for the final binary",
        "2.".bright_blue()
    );
    println!(
        "  {} Run {} to start fresh",
        "3.".bright_blue(),
        "batuta reset".cyan()
    );
    println!();

    Ok(())
}

// ============================================================================
// Report Command
// ============================================================================

pub fn cmd_report(output: PathBuf, format: ReportFormat) -> anyhow::Result<()> {
    println!(
        "{}",
        "📊 Generating migration report...".bright_cyan().bold()
    );
    println!();

    // Load workflow state
    let state_file = super::get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if any work has been done
    let has_started = state
        .phases
        .values()
        .any(|info| info.status != PhaseStatus::NotStarted);
    if !has_started {
        println!("{}", "⚠️  No workflow data found!".yellow().bold());
        println!();
        println!(
            "Run {} first to generate analysis data.",
            "batuta analyze".cyan()
        );
        println!();
        return Ok(());
    }

    // Load or create analysis
    let config_path = PathBuf::from("batuta.toml");
    let analysis = if config_path.exists() {
        let config = BatutaConfig::load(&config_path)?;
        analyze_project(&config.source.path, true, true, true)?
    } else {
        // Use current directory if no config
        analyze_project(&PathBuf::from("."), true, true, true)?
    };

    // Create report
    let project_name = analysis
        .root_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let migration_report = report::MigrationReport::new(project_name, analysis, state);

    // Convert format enum
    let report_format = match format {
        ReportFormat::Html => report::ReportFormat::Html,
        ReportFormat::Markdown => report::ReportFormat::Markdown,
        ReportFormat::Json => report::ReportFormat::Json,
        ReportFormat::Text => report::ReportFormat::Text,
    };

    // Save report
    migration_report.save(&output, report_format)?;

    println!(
        "{}",
        "✅ Report generated successfully!".bright_green().bold()
    );
    println!();
    println!("{}: {:?}", "Output file".bold(), output);
    println!("{}: {:?}", "Format".bold(), format);
    println!();

    // Show preview for text-based formats
    if matches!(format, ReportFormat::Text | ReportFormat::Markdown) {
        println!("{}", "Preview (first 20 lines):".dimmed());
        println!("{}", "─".repeat(80).dimmed());
        let content = std::fs::read_to_string(&output)?;
        for line in content.lines().take(20) {
            println!("{}", line.dimmed());
        }
        if content.lines().count() > 20 {
            println!("{}", "...".dimmed());
        }
        println!("{}", "─".repeat(80).dimmed());
        println!();
    }

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Open the report to view detailed analysis",
        "1.".bright_blue()
    );
    if matches!(format, ReportFormat::Html) {
        println!(
            "  {} Open in browser: file://{}",
            "2.".bright_blue(),
            output.canonicalize()?.display()
        );
    }
    println!();

    Ok(())
}
