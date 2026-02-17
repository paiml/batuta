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
            println!("{}: {}", "Recommended transpiler".bold(), t.bright_green());
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
        println!("{}", "⚠️  Analysis phase not completed!".yellow().bold());
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

fn setup_transpiler(config: &BatutaConfig) -> anyhow::Result<(ToolRegistry, Language, ToolInfo)> {
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
        transpiler
            .version
            .as_ref()
            .unwrap_or(&"unknown".to_string())
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
    println!("  {} Source: {:?}", "•".bright_blue(), config.source.path);
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
        run_ruchy_repl(config)?;
    }

    Ok(())
}

fn run_ruchy_repl(config: &BatutaConfig) -> anyhow::Result<()> {
    // Check if ruchy is available
    let tools = ToolRegistry::detect();
    if tools.ruchy.is_none() {
        println!("{} Ruchy is not installed", "✗".red());
        println!();
        println!("Install Ruchy to use the interactive REPL:");
        println!("  {}", "cargo install ruchy".cyan());
        anyhow::bail!("Ruchy not found in PATH");
    }

    print_repl_banner(config);

    let stdin = std::io::stdin();
    let mut buffer = String::new();

    loop {
        print_repl_prompt(&buffer);

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }

        match process_repl_line(&line, &mut buffer, config) {
            ReplAction::Continue => {}
            ReplAction::Quit => break,
        }
    }

    println!();
    println!("{}", "Ruchy REPL session ended.".dimmed());
    Ok(())
}

enum ReplAction {
    Continue,
    Quit,
}

fn print_repl_banner(config: &BatutaConfig) {
    println!("{}", "🔬 Ruchy REPL".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "  Output dir: {}",
        config.transpilation.output_dir.display()
    );
    println!(
        "  Strictness: {}",
        config
            .transpilation
            .ruchy_strictness
            .as_deref()
            .unwrap_or("gradual")
    );
    println!();
    println!("  Type Ruchy code, then press Enter twice to execute.");
    println!(
        "  Commands: {} {} {}",
        ":help".cyan(),
        ":clear".cyan(),
        ":quit".cyan()
    );
    println!("{}", "─".repeat(50).dimmed());
    println!();
}

fn print_repl_prompt(buffer: &str) {
    if buffer.is_empty() {
        eprint!("{} ", "ruchy>".bright_blue());
    } else {
        eprint!("{} ", "   ..>".dimmed());
    }
}

fn process_repl_line(line: &str, buffer: &mut String, config: &BatutaConfig) -> ReplAction {
    let trimmed = line.trim();

    // Handle REPL commands (only when buffer is empty)
    if buffer.is_empty() {
        match trimmed {
            ":quit" | ":q" | ":exit" => return ReplAction::Quit,
            ":help" | ":h" => {
                print_repl_help();
                return ReplAction::Continue;
            }
            ":clear" | ":c" => {
                buffer.clear();
                println!("{}", "Buffer cleared.".dimmed());
                return ReplAction::Continue;
            }
            _ => {}
        }
    }

    // Empty line on non-empty buffer = execute
    if trimmed.is_empty() && !buffer.is_empty() {
        execute_repl_snippet(buffer, config);
        buffer.clear();
        return ReplAction::Continue;
    }

    // Accumulate code
    if !trimmed.is_empty() {
        buffer.push_str(line);
    }

    ReplAction::Continue
}

fn print_repl_help() {
    println!();
    println!("{}", "Ruchy REPL Commands:".bright_yellow().bold());
    println!("  {} — Show this help", ":help".cyan());
    println!("  {} — Clear the input buffer", ":clear".cyan());
    println!("  {} — Exit the REPL", ":quit".cyan());
    println!();
    println!("{}", "Usage:".bright_yellow().bold());
    println!("  Type code, press Enter twice to transpile and run.");
    println!("  Multi-line input is supported — keep typing until");
    println!("  you enter a blank line.");
    println!();
}

fn execute_repl_snippet(code: &str, config: &BatutaConfig) {
    // Write snippet to a temp file
    let tmp_dir = std::env::temp_dir();
    let snippet_path = tmp_dir.join("batuta_repl_snippet.rcy");
    if let Err(e) = std::fs::write(&snippet_path, code) {
        println!("{} Failed to write snippet: {}", "✗".red(), e);
        return;
    }

    // Build ruchy args
    let snippet_str = snippet_path.to_string_lossy().to_string();
    let strictness = config
        .transpilation
        .ruchy_strictness
        .as_deref()
        .unwrap_or("gradual");
    let args: Vec<&str> = vec!["run", "--strictness", strictness, &snippet_str];

    println!("{}", "─".repeat(50).dimmed());
    match tools::run_tool("ruchy", &args, None) {
        Ok(output) => {
            if output.trim().is_empty() {
                println!("{}", "(no output)".dimmed());
            } else {
                println!("{}", output);
            }
        }
        Err(e) => {
            println!("{} {}", "Error:".red(), e);
        }
    }
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Clean up
    let _ = std::fs::remove_file(&snippet_path);
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
// Optimize Helpers
// ============================================================================

/// A detected compute pattern in transpiled source code.
struct ComputePattern {
    file: String,
    kind: crate::backend::OpComplexity,
    description: String,
}

/// Scan transpiled Rust files for compute-intensive patterns.
fn scan_optimization_targets(output_dir: &Path) -> Vec<ComputePattern> {
    use crate::backend::OpComplexity;

    let mut patterns = Vec::new();
    let rs_files = collect_rs_files(output_dir);

    for path in &rs_files {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let file = path
            .strip_prefix(output_dir)
            .unwrap_or(path)
            .display()
            .to_string();

        // High complexity: matrix operations
        for kw in &[
            "matmul",
            "matrix_multiply",
            "gemm",
            "dot_product",
            "convolution",
        ] {
            if content.contains(kw) {
                patterns.push(ComputePattern {
                    file: file.clone(),
                    kind: OpComplexity::High,
                    description: format!("matrix/convolution op: {}", kw),
                });
            }
        }

        // Medium complexity: reductions and aggregations
        for kw in &[".sum()", ".product()", ".fold(", "reduce(", ".norm("] {
            if content.contains(kw) {
                patterns.push(ComputePattern {
                    file: file.clone(),
                    kind: OpComplexity::Medium,
                    description: format!("reduction op: {}", kw.trim_matches('.')),
                });
            }
        }

        // Low complexity: element-wise operations in loops
        if content.contains(".iter()") && (content.contains(".map(") || content.contains(".zip(")) {
            patterns.push(ComputePattern {
                file: file.clone(),
                kind: OpComplexity::Low,
                description: "element-wise iter/map/zip pattern".to_string(),
            });
        }
    }

    patterns
}

/// Collect all .rs files under a directory.
fn collect_rs_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_rs_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    files
}

/// Run MoE backend analysis on detected compute patterns.
fn run_moe_analysis(
    enable_gpu: bool,
    enable_simd: bool,
    gpu_threshold: usize,
    patterns: &[ComputePattern],
) -> Vec<String> {
    use crate::pipeline::OptimizationStage;

    let stage = OptimizationStage::new(enable_gpu, enable_simd, gpu_threshold);
    let mut recommendations = stage.analyze_optimizations();

    // Add per-file recommendations based on detected patterns
    for pat in patterns {
        let backend = stage.backend_selector.select_with_moe(pat.kind, 10_000);
        recommendations.push(format!(
            "{}: {} → {} backend",
            pat.file, pat.description, backend
        ));
    }

    recommendations
}

/// Apply profile-specific [profile.release] settings to Cargo.toml.
fn apply_profile_optimizations(
    cargo_toml: &Path,
    profile: OptimizationProfile,
) -> anyhow::Result<Vec<String>> {
    let content = std::fs::read_to_string(cargo_toml)?;
    let mut applied = Vec::new();

    let (opt_level, lto, codegen_units, strip) = match profile {
        OptimizationProfile::Fast => ("2", "false", "16", "none"),
        OptimizationProfile::Balanced => ("3", "thin", "4", "none"),
        OptimizationProfile::Aggressive => ("3", "true", "1", "symbols"),
    };

    // Only append profile section if not already present
    if content.contains("[profile.release]") {
        applied.push(format!(
            "[profile.release] already exists — manual review recommended (profile: {:?})",
            profile
        ));
        return Ok(applied);
    }

    let section = format!(
        "\n[profile.release]\nopt-level = \"{}\"\nlto = \"{}\"\ncodegen-units = {}\nstrip = \"{}\"\n",
        opt_level, lto, codegen_units, strip
    );
    let mut new_content = content;
    new_content.push_str(&section);
    std::fs::write(cargo_toml, new_content)?;

    applied.push(format!("opt-level = \"{}\"", opt_level));
    applied.push(format!("lto = \"{}\"", lto));
    applied.push(format!("codegen-units = {}", codegen_units));
    applied.push(format!("strip = \"{}\"", strip));

    Ok(applied)
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

    // Load project config to find the transpiled output directory
    let config_path = PathBuf::from("batuta.toml");
    let config = if config_path.exists() {
        BatutaConfig::load(&config_path)?
    } else {
        BatutaConfig::default()
    };

    let output_dir = &config.transpilation.output_dir;
    if !output_dir.exists() {
        println!(
            "{} Output directory not found: {}",
            "✗".red(),
            output_dir.display()
        );
        state.fail_phase(
            WorkflowPhase::Optimization,
            format!("Output directory not found: {}", output_dir.display()),
        );
        state.save(&state_file)?;
        anyhow::bail!(
            "Transpiled output directory not found: {}",
            output_dir.display()
        );
    }

    // Scan transpiled source and run MoE analysis
    let patterns = scan_optimization_targets(output_dir);
    let recommendations = run_moe_analysis(enable_gpu, enable_simd, gpu_threshold, &patterns);

    // Display MoE recommendations
    println!("{}", "MoE Backend Analysis:".bright_yellow().bold());
    if recommendations.is_empty() {
        println!("  {} No compute-intensive patterns detected", "•".dimmed());
    } else {
        for rec in &recommendations {
            println!("  {} {}", "→".bright_blue(), rec);
        }
    }
    println!();

    // Apply profile-specific Cargo.toml optimizations
    let cargo_toml = output_dir.join("Cargo.toml");
    if cargo_toml.exists() {
        let applied = apply_profile_optimizations(&cargo_toml, profile)?;
        println!("{}", "Cargo Profile Optimizations:".bright_yellow().bold());
        for opt in &applied {
            println!("  {} {}", "✓".bright_green(), opt);
        }
        println!();
    }

    // Summary
    println!(
        "{} Analyzed {} source patterns, generated {} recommendations",
        "✅".bright_green(),
        patterns.len(),
        recommendations.len()
    );

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

    if diff_output && !run_output_diff() {
        validation_passed = false;
    }

    if run_original_tests && !run_transpiled_tests() {
        validation_passed = false;
    }

    if benchmark && !run_performance_benchmark() {
        validation_passed = false;
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

/// Run output diff comparison between original and transpiled binaries.
/// Returns true if outputs match or binaries are not found.
fn run_output_diff() -> bool {
    println!("{}", "📊 Output comparison:".bright_cyan());

    let original_binary = Path::new("./original_binary");
    let transpiled_binary = Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for comparison".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    println!("  {} Running original...", "•".bright_blue());
    let original_out = std::process::Command::new(original_binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    println!("  {} Running transpiled...", "•".bright_blue());
    let transpiled_out = std::process::Command::new(transpiled_binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    match (original_out, transpiled_out) {
        (Ok(orig), Ok(trans)) => {
            let orig_stdout = String::from_utf8_lossy(&orig.stdout);
            let trans_stdout = String::from_utf8_lossy(&trans.stdout);

            if orig_stdout == trans_stdout {
                println!(
                    "{}",
                    "  ✅ Outputs match - functional equivalence verified".green()
                );
                println!();
                true
            } else {
                println!("{}", "  ❌ Outputs differ:".red());
                show_output_diff(&orig_stdout, &trans_stdout);
                println!();
                false
            }
        }
        (Err(e), _) => {
            println!(
                "{}",
                format!("  ❌ Failed to run original binary: {e}").red()
            );
            println!();
            false
        }
        (_, Err(e)) => {
            println!(
                "{}",
                format!("  ❌ Failed to run transpiled binary: {e}").red()
            );
            println!();
            false
        }
    }
}

/// Show a simple line-by-line diff between two outputs.
fn show_output_diff(original: &str, transpiled: &str) {
    let orig_lines: Vec<&str> = original.lines().collect();
    let trans_lines: Vec<&str> = transpiled.lines().collect();
    let max = orig_lines.len().max(trans_lines.len()).min(20);

    for i in 0..max {
        let orig_line = orig_lines.get(i).unwrap_or(&"");
        let trans_line = trans_lines.get(i).unwrap_or(&"");
        if orig_line != trans_line {
            println!("    {} {}", "- ".red(), orig_line);
            println!("    {} {}", "+ ".green(), trans_line);
        }
    }
    if orig_lines.len().max(trans_lines.len()) > 20 {
        println!(
            "    ... (truncated, {} total lines)",
            orig_lines.len().max(trans_lines.len())
        );
    }
}

/// Run `cargo test` in the transpiled output directory. Returns true if tests pass.
fn run_transpiled_tests() -> bool {
    println!(
        "{}",
        "🧪 Running test suite on transpiled code:".bright_cyan()
    );

    let config_path = PathBuf::from("batuta.toml");
    let config = if config_path.exists() {
        match BatutaConfig::load(&config_path) {
            Ok(c) => c,
            Err(e) => {
                println!("{}", format!("  ❌ Failed to load config: {e}").red());
                println!();
                return false;
            }
        }
    } else {
        BatutaConfig::default()
    };

    let output_dir = &config.transpilation.output_dir;
    if !output_dir.join("Cargo.toml").exists() {
        println!(
            "{}",
            format!("  ⚠️  No Cargo.toml in {}", output_dir.display()).yellow()
        );
        println!("     Run {} first.", "batuta build".cyan());
        println!();
        return true;
    }

    println!(
        "  {} Running: cargo test in {}",
        "•".bright_blue(),
        output_dir.display()
    );

    let status = std::process::Command::new("cargo")
        .arg("test")
        .current_dir(output_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match status {
        Ok(s) if s.success() => {
            println!();
            println!("{}", "  ✅ All tests pass on transpiled code".green());
            println!();
            true
        }
        Ok(s) => {
            let code = s.code().map_or("signal".to_string(), |c| c.to_string());
            println!();
            println!("{}", format!("  ❌ Tests failed (exit {})", code).red());
            println!();
            false
        }
        Err(e) => {
            println!("{}", format!("  ❌ Failed to run cargo test: {e}").red());
            println!();
            false
        }
    }
}

/// Run performance benchmarks comparing original vs transpiled. Returns true always (informational).
fn run_performance_benchmark() -> bool {
    println!("{}", "⚡ Performance benchmarking:".bright_cyan());

    let original_binary = Path::new("./original_binary");
    let transpiled_binary = Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for benchmarking".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    let iterations = 3;
    println!(
        "  {} Running {} iterations each...",
        "•".bright_blue(),
        iterations
    );

    let orig_time = time_binary_avg(original_binary, iterations);
    let trans_time = time_binary_avg(transpiled_binary, iterations);

    match (orig_time, trans_time) {
        (Some(orig_ms), Some(trans_ms)) => {
            println!();
            println!("  Original:   {:.1}ms avg", orig_ms);
            println!("  Transpiled: {:.1}ms avg", trans_ms);
            if trans_ms > 0.0 {
                let speedup = orig_ms / trans_ms;
                if speedup >= 1.0 {
                    println!("  Speedup:    {:.2}x {}", speedup, "faster".green());
                } else {
                    println!("  Speedup:    {:.2}x {}", speedup, "slower".red());
                }
            }
            println!();
            true
        }
        _ => {
            println!("{}", "  ❌ Failed to benchmark binaries".red());
            println!();
            false
        }
    }
}

/// Time a binary over N iterations, returning average milliseconds.
fn time_binary_avg(binary: &Path, iterations: u32) -> Option<f64> {
    let mut total = std::time::Duration::ZERO;
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let status = std::process::Command::new(binary)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        total += start.elapsed();
    }
    Some(total.as_secs_f64() * 1000.0 / f64::from(iterations))
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

    println!("  {} Project: {}", "•".bright_blue(), output_dir.display());
    println!();

    // Execute cargo build in the transpiled project
    match run_cargo_build(
        output_dir,
        release,
        target.as_deref(),
        wasm,
        &config.build.cargo_flags,
    ) {
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
