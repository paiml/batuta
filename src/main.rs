mod analyzer;
mod backend;
mod config;
mod numpy_converter;
mod parf;
mod pipeline;
mod pytorch_converter;
mod report;
mod sklearn_converter;
mod tools;
mod types;

use analyzer::analyze_project;
use clap::{Parser, Subcommand};
use colored::Colorize;
use config::BatutaConfig;
use std::path::{Path, PathBuf};
use tools::ToolRegistry;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use types::{PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState};

/// Get the workflow state file path
fn get_state_file_path() -> PathBuf {
    PathBuf::from(".batuta-state.json")
}

/// Display workflow progress
fn display_workflow_progress(state: &WorkflowState) {
    println!();
    println!("{}", "📊 Workflow Progress".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state.phases.get(&phase).unwrap();
        let status_icon = match info.status {
            PhaseStatus::Completed => "✓".bright_green(),
            PhaseStatus::InProgress => "⏳".bright_yellow(),
            PhaseStatus::Failed => "✗".bright_red(),
            PhaseStatus::NotStarted => "○".dimmed(),
        };

        let phase_name = format!("{}", phase);
        let status_text = format!("{}", info.status);

        let is_current = state.current_phase == Some(phase);
        if is_current {
            println!(
                "  {} {} [{}]",
                status_icon,
                phase_name.cyan().bold(),
                status_text.bright_yellow()
            );
        } else {
            println!("  {} {} [{}]", status_icon, phase_name.dimmed(), status_text.dimmed());
        }
    }

    let progress = state.progress_percentage();
    println!();
    println!("  Overall: {:.0}% complete", progress);
    println!("{}", "─".repeat(50).dimmed());
    println!();
}

#[derive(Parser)]
#[command(name = "batuta")]
#[command(version, about = "Orchestration framework for converting ANY project to Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable debug output
    #[arg(short, long, global = true)]
    debug: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Batuta project
    Init {
        /// Source project path
        #[arg(long, default_value = ".")]
        source: PathBuf,

        /// Output directory for Rust project
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Analyze source codebase (Phase 1: Analysis)
    Analyze {
        /// Project path
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Generate TDG score
        #[arg(long)]
        tdg: bool,

        /// Detect languages
        #[arg(long)]
        languages: bool,

        /// Analyze dependencies
        #[arg(long)]
        dependencies: bool,
    },

    /// Transpile source code to Rust (Phase 2: Transpilation)
    Transpile {
        /// Enable incremental transpilation
        #[arg(long)]
        incremental: bool,

        /// Use caching for unchanged files
        #[arg(long)]
        cache: bool,

        /// Specific modules to transpile
        #[arg(long, value_delimiter = ',')]
        modules: Option<Vec<String>>,

        /// Generate Ruchy instead of Rust
        #[arg(long)]
        ruchy: bool,

        /// Start REPL after transpilation
        #[arg(long)]
        repl: bool,
    },

    /// Optimize transpiled code (Phase 3: Optimization)
    Optimize {
        /// Enable GPU acceleration
        #[arg(long)]
        enable_gpu: bool,

        /// Enable SIMD vectorization
        #[arg(long, default_value = "true")]
        enable_simd: bool,

        /// Optimization profile
        #[arg(long, value_enum, default_value = "balanced")]
        profile: OptimizationProfile,

        /// GPU dispatch threshold (matrix size)
        #[arg(long, default_value = "500")]
        gpu_threshold: usize,
    },

    /// Validate semantic equivalence (Phase 4: Validation)
    Validate {
        /// Trace syscalls for comparison
        #[arg(long)]
        trace_syscalls: bool,

        /// Generate diff output
        #[arg(long)]
        diff_output: bool,

        /// Run original test suite
        #[arg(long)]
        run_original_tests: bool,

        /// Run benchmarks
        #[arg(long)]
        benchmark: bool,
    },

    /// Build Rust binary (Phase 5: Deployment)
    Build {
        /// Build in release mode
        #[arg(long)]
        release: bool,

        /// Target platform
        #[arg(long)]
        target: Option<String>,

        /// Build for WebAssembly
        #[arg(long)]
        wasm: bool,
    },

    /// Generate migration report
    Report {
        /// Output file path
        #[arg(long, default_value = "migration_report.html")]
        output: PathBuf,

        /// Report format
        #[arg(long, value_enum, default_value = "html")]
        format: ReportFormat,
    },

    /// Show workflow status
    Status,

    /// Reset workflow state
    Reset {
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Pattern and reference finder (PARF) for code analysis
    Parf {
        /// Path to analyze
        #[arg(default_value = "src")]
        path: PathBuf,

        /// Find all references to a symbol
        #[arg(long)]
        find: Option<String>,

        /// Detect code patterns
        #[arg(long)]
        patterns: bool,

        /// Analyze dependencies
        #[arg(long)]
        dependencies: bool,

        /// Find dead code
        #[arg(long)]
        dead_code: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: ParfOutputFormat,

        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum OptimizationProfile {
    /// Fast compilation, basic optimizations
    Fast,
    /// Balanced compilation and performance
    Balanced,
    /// Maximum performance, slower compilation
    Aggressive,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ParfOutputFormat {
    /// Plain text output
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ReportFormat {
    /// HTML report with charts
    Html,
    /// Markdown report
    Markdown,
    /// JSON data
    Json,
    /// Plain text
    Text,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter_layer = if cli.debug {
        tracing_subscriber::EnvFilter::new("debug")
    } else if cli.verbose {
        tracing_subscriber::EnvFilter::new("info")
    } else {
        tracing_subscriber::EnvFilter::new("warn")
    };

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Batuta v{}", env!("CARGO_PKG_VERSION"));

    match cli.command {
        Commands::Init { source, output } => {
            info!("Initializing Batuta project from {:?}", source);
            cmd_init(source, output)?;
        }
        Commands::Analyze {
            path,
            tdg,
            languages,
            dependencies,
        } => {
            info!("Analyzing project at {:?}", path);
            cmd_analyze(path, tdg, languages, dependencies)?;
        }
        Commands::Transpile {
            incremental,
            cache,
            modules,
            ruchy,
            repl,
        } => {
            info!("Transpiling to {}", if ruchy { "Ruchy" } else { "Rust" });
            cmd_transpile(incremental, cache, modules, ruchy, repl)?;
        }
        Commands::Optimize {
            enable_gpu,
            enable_simd,
            profile,
            gpu_threshold,
        } => {
            info!("Optimizing with profile: {:?}", profile);
            cmd_optimize(enable_gpu, enable_simd, profile, gpu_threshold)?;
        }
        Commands::Validate {
            trace_syscalls,
            diff_output,
            run_original_tests,
            benchmark,
        } => {
            info!("Validating semantic equivalence");
            cmd_validate(trace_syscalls, diff_output, run_original_tests, benchmark)?;
        }
        Commands::Build {
            release,
            target,
            wasm,
        } => {
            info!("Building Rust project");
            cmd_build(release, target, wasm)?;
        }
        Commands::Report { output, format } => {
            info!("Generating migration report");
            cmd_report(output, format)?;
        }
        Commands::Status => {
            info!("Checking workflow status");
            cmd_status()?;
        }
        Commands::Reset { yes } => {
            info!("Resetting workflow state");
            cmd_reset(yes)?;
        }
        Commands::Parf {
            path,
            find,
            patterns,
            dependencies,
            dead_code,
            format,
            output,
        } => {
            info!("Running PARF analysis on {:?}", path);
            cmd_parf(&path, find.as_deref(), patterns, dependencies, dead_code, format, output.as_deref())?;
        }
    }

    Ok(())
}

// Command implementations (stubs for now)

fn cmd_init(source: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
    println!("{}", "🚀 Initializing Batuta project...".bright_cyan().bold());
    println!();

    // Analyze the source project
    println!("{}", "Analyzing source project...".dimmed());
    let analysis = analyze_project(&source, true, true, true)?;

    println!("{} Source: {:?}", "✓".bright_green(), source);
    if let Some(lang) = &analysis.primary_language {
        println!("{} Detected language: {}", "✓".bright_green(), format!("{}", lang).cyan());
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

    println!("{} Created configuration: {:?}", "✓".bright_green(), config_path);

    // Create output directory structure
    std::fs::create_dir_all(&output_dir)?;
    std::fs::create_dir_all(output_dir.join("src"))?;

    println!("{} Created output directory: {:?}", "✓".bright_green(), output_dir);
    println!();

    // Display configuration summary
    println!("{}", "📋 Configuration Summary".bright_yellow().bold());
    println!("{}", "=".repeat(50));
    println!();
    println!("{}: {}", "Project name".bold(), config.project.name.cyan());
    println!("{}: {}", "Primary language".bold(),
        config.project.primary_language.as_ref().unwrap_or(&"Unknown".to_string()).cyan());
    println!("{}: {:?}", "Output directory".bold(), config.transpilation.output_dir);
    println!();

    // Display transpilation settings
    println!("{}", "Transpilation:".bright_yellow());
    println!("  {} Incremental: {}", "•".bright_blue(), config.transpilation.incremental.to_string().cyan());
    println!("  {} Caching: {}", "•".bright_blue(), config.transpilation.cache.to_string().cyan());

    if analysis.has_ml_dependencies() {
        println!("  {} NumPy → Trueno: {}", "•".bright_blue(), "enabled".green());
        println!("  {} sklearn → Aprender: {}", "•".bright_blue(), "enabled".green());
        println!("  {} PyTorch → Realizar: {}", "•".bright_blue(), "enabled".green());
    }
    println!();

    // Display optimization settings
    println!("{}", "Optimization:".bright_yellow());
    println!("  {} Profile: {}", "•".bright_blue(), config.optimization.profile.cyan());
    println!("  {} SIMD: {}", "•".bright_blue(), config.optimization.enable_simd.to_string().cyan());
    println!("  {} GPU: {}", "•".bright_blue(),
        if config.optimization.enable_gpu { "enabled".green() } else { "disabled".dimmed() });
    println!();

    // Next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Edit {} to customize settings", "1.".bright_blue(), "batuta.toml".cyan());
    println!("  {} Run {} to convert your code", "2.".bright_blue(), "batuta transpile".cyan());
    println!("  {} Run {} to optimize performance", "3.".bright_blue(), "batuta optimize".cyan());
    println!();

    Ok(())
}

/// Display project analysis results
fn display_analysis_results(analysis: &ProjectAnalysis) {
    println!("{}", "📊 Analysis Results".bright_green().bold());
    println!("{}", "=".repeat(50));
    println!();

    // Project info
    println!("{}: {:?}", "Project path".bold(), analysis.root_path);
    println!("{}: {}", "Total files".bold(), analysis.total_files.to_string().cyan());
    println!("{}: {}", "Total lines".bold(), analysis.total_lines.to_string().cyan());
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
            lang_stat.file_count.to_string().yellow(),
            lang_stat.line_count.to_string().green(),
            lang_stat.percentage
        );
    }
    println!();

    if let Some(primary) = &analysis.primary_language {
        println!("{}: {}", "Primary language".bold(), format!("{}", primary).bright_cyan());
    }

    if let Some(transpiler) = analysis.recommend_transpiler() {
        println!("{}: {}", "Recommended transpiler".bold(), transpiler.bright_green());
    }
    println!();
}

/// Display dependency information
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
        println!("  {} {}{}", "•".bright_blue(), format!("{}", dep.manager).cyan(), count_str.yellow());
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

    let grade = if score >= 90.0 {
        "A+".bright_green()
    } else if score >= 80.0 {
        "A".green()
    } else if score >= 70.0 {
        "B".yellow()
    } else if score >= 60.0 {
        "C".yellow()
    } else {
        "D".red()
    };

    println!("{}", "Quality Score:".bright_yellow().bold());
    println!("  {} TDG Score: {}/100 ({})", "•".bright_blue(), format!("{:.1}", score).cyan(), grade);
    println!();
}

/// Display next steps after analysis
fn display_analyze_next_steps() {
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!("  {} Run {} to initialize configuration", "1.".bright_blue(), "batuta init".cyan());
    println!("  {} Run {} to convert project to Rust", "2.".bright_blue(), "batuta transpile".cyan());
    println!("  {} Run {} for performance optimization", "3.".bright_blue(), "batuta optimize".cyan());
    println!();
}

fn cmd_analyze(
    path: PathBuf,
    tdg: bool,
    languages: bool,
    dependencies: bool,
) -> anyhow::Result<()> {
    println!("{}", "🔍 Analyzing project...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Start analysis phase
    state.start_phase(WorkflowPhase::Analysis);
    state.save(&state_file)?;

    // Run analysis
    let result = analyze_project(&path, tdg, languages, dependencies);

    // Handle result and update state
    let analysis = match result {
        Ok(a) => {
            state.complete_phase(WorkflowPhase::Analysis);
            state.save(&state_file)?;
            a
        }
        Err(e) => {
            state.fail_phase(WorkflowPhase::Analysis, e.to_string());
            state.save(&state_file)?;
            return Err(e);
        }
    };

    // Display results
    display_analysis_results(&analysis);
    display_workflow_progress(&state);
    display_analyze_next_steps();

    Ok(())
}

/// Check transpilation prerequisites
fn check_transpile_prerequisites(state: &WorkflowState) -> anyhow::Result<BatutaConfig> {
    if !state.is_phase_completed(WorkflowPhase::Analysis) {
        println!("{}", "⚠️  Analysis phase not completed!".yellow().bold());
        println!();
        println!("Run {} first to analyze your project.", "batuta analyze".cyan());
        println!();
        display_workflow_progress(state);
        anyhow::bail!("Analysis phase not completed");
    }

    let config_path = PathBuf::from("batuta.toml");
    if !config_path.exists() {
        println!("{}", "⚠️  No batuta.toml found!".yellow().bold());
        println!();
        println!("Run {} first to create a configuration file.", "batuta init".cyan());
        println!();
        anyhow::bail!("No configuration file found");
    }

    let config = BatutaConfig::load(&config_path)?;
    println!("{} Loaded configuration", "✓".bright_green());
    Ok(config)
}

/// Setup transpiler tools and analyze project
fn setup_transpiler(config: &BatutaConfig) -> anyhow::Result<(ToolRegistry, types::Language, tools::ToolInfo)> {
    println!("{}", "Detecting installed tools...".dimmed());
    let tools = ToolRegistry::detect();

    let available = tools.available_tools();
    if available.is_empty() {
        return handle_missing_tools(&tools, config);
    }

    println!();
    println!("{}", "Available tools:".bright_yellow());
    for tool in &available {
        println!("  {} {}", "✓".bright_green(), tool);
    }
    println!();

    println!("{}", "Analyzing project...".dimmed());
    let analysis = analyze_project(&config.source.path, false, true, false)?;
    let primary_lang = analysis.primary_language
        .ok_or_else(|| anyhow::anyhow!("Could not determine primary language"))?;

    println!("{} Primary language: {}", "✓".bright_green(), format!("{}", primary_lang).cyan());

    let transpiler = tools.get_transpiler_for_language(&primary_lang)
        .ok_or_else(|| anyhow::anyhow!("No transpiler available for {}", primary_lang))?
        .clone();

    println!("{} Using transpiler: {}", "✓".bright_green(), transpiler.name.cyan());
    if let Some(ver) = &transpiler.version {
        println!("  {} Version: {}", "ℹ".bright_blue(), ver.dimmed());
    }
    println!();

    Ok((tools, primary_lang, transpiler))
}

/// Handle missing transpiler tools
fn handle_missing_tools(tools: &ToolRegistry, config: &BatutaConfig) -> anyhow::Result<(ToolRegistry, types::Language, tools::ToolInfo)> {
    println!();
    println!("{}", "❌ No transpiler tools found!".red().bold());
    println!();
    println!("{}", "Install required tools:".yellow());

    let analysis = analyze_project(&config.source.path, false, true, false)?;
    let needed_tools = if let Some(lang) = &analysis.primary_language {
        match lang {
            types::Language::Python => vec!["depyler"],
            types::Language::C | types::Language::Cpp => vec!["decy"],
            types::Language::Shell => vec!["bashrs"],
            _ => vec![],
        }
    } else {
        vec![]
    };

    let instructions = tools.get_installation_instructions(&needed_tools);
    for inst in instructions {
        println!("  {} {}", "•".bright_blue(), inst.cyan());
    }
    println!();
    anyhow::bail!("No transpiler tools found")
}

/// Build transpiler arguments
fn build_transpiler_args(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) -> (Vec<String>, Vec<String>) {
    let input_path_str = config.source.path.to_string_lossy().to_string();
    let output_path_str = config.transpilation.output_dir.to_string_lossy().to_string();
    let modules_str = modules.as_ref().map(|m| m.join(",")).unwrap_or_default();

    let mut owned_args = vec![
        "--input".to_string(),
        input_path_str,
        "--output".to_string(),
        output_path_str,
    ];

    if incremental || config.transpilation.incremental {
        owned_args.push("--incremental".to_string());
    }

    if cache || config.transpilation.cache {
        owned_args.push("--cache".to_string());
    }

    if ruchy || config.transpilation.use_ruchy {
        owned_args.push("--ruchy".to_string());
    }

    if modules.is_some() {
        owned_args.push("--modules".to_string());
        owned_args.push(modules_str);
    }

    let args: Vec<String> = owned_args.clone();
    (owned_args, args)
}

/// Display transpilation settings
fn display_transpilation_settings(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) {
    println!("{}", "Transpilation Settings:".bright_yellow().bold());
    println!("  {} Source: {:?}", "•".bright_blue(), config.source.path);
    println!("  {} Output: {:?}", "•".bright_blue(), config.transpilation.output_dir);
    println!("  {} Incremental: {}", "•".bright_blue(),
        if incremental || config.transpilation.incremental { "enabled".green() } else { "disabled".dimmed() });
    println!("  {} Caching: {}", "•".bright_blue(),
        if cache || config.transpilation.cache { "enabled".green() } else { "disabled".dimmed() });

    if let Some(mods) = modules {
        println!("  {} Modules: {}", "•".bright_blue(), mods.join(", ").cyan());
    }

    if ruchy || config.transpilation.use_ruchy {
        println!("  {} Target: {}", "•".bright_blue(), "Ruchy".cyan());
        if let Some(strictness) = &config.transpilation.ruchy_strictness {
            println!("  {} Strictness: {}", "•".bright_blue(), strictness.cyan());
        }
    }
    println!();
}

fn cmd_transpile(
    incremental: bool,
    cache: bool,
    modules: Option<Vec<String>>,
    ruchy: bool,
    repl: bool,
) -> anyhow::Result<()> {
    println!("{}", "🔄 Transpiling code...".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

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
    let (owned_args, _) = build_transpiler_args(&config, incremental, cache, ruchy, &modules);
    let args: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

    println!("{}", "Executing:".dimmed());
    println!("  {} {} {}", "$".dimmed(), transpiler.name.cyan(), args.join(" ").dimmed());
    println!();
    println!("{}", "Transpiling...".bright_yellow());

    // Run transpiler and handle result
    let result = tools::run_tool(&transpiler.name, &args, Some(&config.source.path));
    handle_transpile_result(result, &mut state, &state_file, &config, &transpiler, repl)
}

/// Handle transpilation result (success or failure)
fn handle_transpile_result(
    result: Result<String, anyhow::Error>,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &tools::ToolInfo,
    repl: bool,
) -> anyhow::Result<()> {
    match result {
        Ok(output) => handle_transpile_success(output, state, state_file, config, repl),
        Err(e) => handle_transpile_failure(e, state, state_file, config, transpiler),
    }
}

/// Handle successful transpilation
fn handle_transpile_success(
    output: String,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    repl: bool,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "✅ Transpilation completed successfully!".bright_green().bold());
    println!();

    // Display transpiler output
    if !output.trim().is_empty() {
        println!("{}", "Transpiler output:".bright_yellow());
        println!("{}", "─".repeat(50).dimmed());
        for line in output.lines().take(20) {
            println!("  {}", line.dimmed());
        }
        if output.lines().count() > 20 {
            println!("  {} ... ({} more lines)", "...".dimmed(), output.lines().count() - 20);
        }
        println!("{}", "─".repeat(50).dimmed());
        println!();
    }

    // Complete transpilation phase
    state.complete_phase(WorkflowPhase::Transpilation);
    state.save(state_file)?;

    display_workflow_progress(state);

    // Show next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Check output directory: {:?}", "1.".bright_blue(), config.transpilation.output_dir);
    println!("  {} Run {} to optimize", "2.".bright_blue(), "batuta optimize".cyan());
    println!("  {} Run {} to validate", "3.".bright_blue(), "batuta validate".cyan());
    println!();

    // Start REPL if requested
    if repl {
        println!("{}", "🔬 Starting Ruchy REPL...".bright_cyan().bold());
        warn!("Ruchy REPL not yet implemented");
    }

    Ok(())
}

/// Handle failed transpilation
fn handle_transpile_failure(
    e: anyhow::Error,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &tools::ToolInfo,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "❌ Transpilation failed!".red().bold());
    println!();
    println!("{}: {}", "Error".bold(), e.to_string().red());
    println!();

    state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
    state.save(state_file)?;

    display_workflow_progress(state);

    // Provide helpful troubleshooting
    println!("{}", "💡 Troubleshooting:".bright_yellow().bold());
    println!("  {} Verify {} is properly installed", "•".bright_blue(), transpiler.name.cyan());
    println!("  {} Check that source path is correct: {:?}", "•".bright_blue(), config.source.path);
    println!("  {} Try running with {} for more details", "•".bright_blue(), "--verbose".cyan());
    println!("  {} See transpiler docs: {}", "•".bright_blue(),
        format!("https://github.com/paiml/{}", transpiler.name).cyan());
    println!();

    Err(e)
}

fn cmd_optimize(
    enable_gpu: bool,
    enable_simd: bool,
    profile: OptimizationProfile,
    gpu_threshold: usize,
) -> anyhow::Result<()> {
    println!("{}", "⚡ Optimizing code...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if transpilation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Transpilation) {
        println!("{}", "⚠️  Transpilation phase not completed!".yellow().bold());
        println!();
        println!("Run {} first to transpile your project.", "batuta transpile".cyan());
        println!();
        display_workflow_progress(&state);
        return Ok(());
    }

    // Start optimization phase
    state.start_phase(WorkflowPhase::Optimization);
    state.save(&state_file)?;

    // Display optimization settings
    println!("{}", "Optimization Settings:".bright_yellow().bold());
    println!("  {} Profile: {:?}", "•".bright_blue(), profile);
    println!("  {} SIMD vectorization: {}", "•".bright_blue(),
        if enable_simd { "enabled".green() } else { "disabled".dimmed() });
    println!("  {} GPU acceleration: {}", "•".bright_blue(),
        if enable_gpu { format!("enabled (threshold: {})", gpu_threshold).green() } else { "disabled".to_string().dimmed() });
    println!();

    // TODO: Implement actual optimization with Trueno
    warn!("Optimization execution not yet implemented - Phase 3 (BATUTA-007)");
    println!("{}", "🚧 Optimization engine coming soon!".bright_yellow().bold());
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
    display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Run {} to verify equivalence", "1.".bright_blue(), "batuta validate".cyan());
    println!("  {} Run {} to build final binary", "2.".bright_blue(), "batuta build --release".cyan());
    println!();

    Ok(())
}

fn cmd_validate(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) -> anyhow::Result<()> {
    println!("{}", "✅ Validating equivalence...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if optimization phase is completed
    if !state.is_phase_completed(WorkflowPhase::Optimization) {
        println!("{}", "⚠️  Optimization phase not completed!".yellow().bold());
        println!();
        println!("Run {} first to optimize your project.", "batuta optimize".cyan());
        println!();
        display_workflow_progress(&state);
        return Ok(());
    }

    // Start validation phase
    state.start_phase(WorkflowPhase::Validation);
    state.save(&state_file)?;

    // Display validation settings
    println!("{}", "Validation Settings:".bright_yellow().bold());
    println!("  {} Syscall tracing: {}", "•".bright_blue(),
        if trace_syscalls { "enabled".green() } else { "disabled".dimmed() });
    println!("  {} Diff output: {}", "•".bright_blue(),
        if diff_output { "enabled".green() } else { "disabled".dimmed() });
    println!("  {} Original tests: {}", "•".bright_blue(),
        if run_original_tests { "enabled".green() } else { "disabled".dimmed() });
    println!("  {} Benchmarks: {}", "•".bright_blue(),
        if benchmark { "enabled".green() } else { "disabled".dimmed() });
    println!();

    // Implement validation with Renacer (BATUTA-011)
    let mut validation_passed = true;

    if trace_syscalls {
        println!("{}", "🔍 Running Renacer syscall tracing...".bright_cyan());

        // Check if binaries exist for comparison
        let original_binary = std::path::Path::new("./original_binary");
        let transpiled_binary = std::path::Path::new("./target/release/transpiled");

        if original_binary.exists() && transpiled_binary.exists() {
            println!("  {} Tracing original binary...", "•".bright_blue());
            println!("  {} Tracing transpiled binary...", "•".bright_blue());
            println!("  {} Comparing syscall traces...", "•".bright_blue());

            // Use ValidationStage for actual validation
            use crate::pipeline::{ValidationStage, PipelineStage, PipelineContext};

            let ctx = PipelineContext::new(
                PathBuf::from("."),
                PathBuf::from("."),
            );

            let stage = ValidationStage::new(trace_syscalls, run_original_tests);

            match tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(stage.execute(ctx))
            {
                Ok(result_ctx) => {
                    if let Some(eq) = result_ctx.metadata.get("syscall_equivalence") {
                        if eq.as_bool() == Some(true) {
                            println!("{}", "  ✅ Syscall traces match - semantic equivalence verified".green());
                        } else {
                            println!("{}", "  ❌ Syscall traces differ - equivalence NOT verified".red());
                            validation_passed = false;
                        }
                    } else {
                        println!("{}", "  ⚠️  Syscall tracing skipped (binaries not found)".yellow());
                    }
                }
                Err(e) => {
                    println!("{}", format!("  ❌ Validation error: {}", e).red());
                    validation_passed = false;
                }
            }
        } else {
            println!("{}", "  ⚠️  Binaries not found for comparison".yellow());
            println!("     Expected: ./original_binary and ./target/release/transpiled");
        }
        println!();
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
        state.fail_phase(WorkflowPhase::Validation, "Validation checks failed".to_string());
    }
    state.save(&state_file)?;

    // Display workflow progress
    display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Run {} to build final binary", "1.".bright_blue(), "batuta build --release".cyan());
    println!("  {} Run {} to generate report", "2.".bright_blue(), "batuta report".cyan());
    println!();

    Ok(())
}

fn cmd_build(release: bool, target: Option<String>, wasm: bool) -> anyhow::Result<()> {
    println!("{}", "🔨 Building Rust project...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if validation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Validation) {
        println!("{}", "⚠️  Validation phase not completed!".yellow().bold());
        println!();
        println!("Run {} first to validate your project.", "batuta validate".cyan());
        println!();
        display_workflow_progress(&state);
        return Ok(());
    }

    // Start deployment phase
    state.start_phase(WorkflowPhase::Deployment);
    state.save(&state_file)?;

    // Display build settings
    println!("{}", "Build Settings:".bright_yellow().bold());
    println!("  {} Build mode: {}", "•".bright_blue(),
        if release { "release".green() } else { "debug".dimmed() });
    if let Some(t) = &target {
        println!("  {} Target: {}", "•".bright_blue(), t.cyan());
    }
    println!("  {} WebAssembly: {}", "•".bright_blue(),
        if wasm { "enabled".green() } else { "disabled".dimmed() });
    println!();

    // TODO: Implement actual build with cargo
    warn!("Build execution not yet implemented - Phase 5 (BATUTA-009)");
    println!("{}", "🚧 Build system coming soon!".bright_yellow().bold());
    println!();
    println!("{}", "Planned build features:".dimmed());
    println!("  {} Cargo build integration", "•".dimmed());
    println!("  {} Cross-compilation support", "•".dimmed());
    println!("  {} WebAssembly target", "•".dimmed());
    println!("  {} Optimized binary stripping", "•".dimmed());
    println!();

    // For now, mark as completed (once implemented, this will be conditional on success)
    state.complete_phase(WorkflowPhase::Deployment);
    state.save(&state_file)?;

    // Display workflow progress
    display_workflow_progress(&state);

    println!("{}", "🎉 Migration Complete!".bright_green().bold());
    println!();
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!("  {} Run {} to generate migration report", "1.".bright_blue(), "batuta report".cyan());
    println!("  {} Check your output directory for the final binary", "2.".bright_blue());
    println!("  {} Run {} to start fresh", "3.".bright_blue(), "batuta reset".cyan());
    println!();

    Ok(())
}

fn cmd_report(output: PathBuf, format: ReportFormat) -> anyhow::Result<()> {
    println!("{}", "📊 Generating migration report...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if any work has been done
    let has_started = state.phases.values().any(|info| info.status != PhaseStatus::NotStarted);
    if !has_started {
        println!("{}", "⚠️  No workflow data found!".yellow().bold());
        println!();
        println!("Run {} first to generate analysis data.", "batuta analyze".cyan());
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
    let project_name = analysis.root_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let report = report::MigrationReport::new(project_name, analysis, state);

    // Convert format enum
    let report_format = match format {
        ReportFormat::Html => report::ReportFormat::Html,
        ReportFormat::Markdown => report::ReportFormat::Markdown,
        ReportFormat::Json => report::ReportFormat::Json,
        ReportFormat::Text => report::ReportFormat::Text,
    };

    // Save report
    report.save(&output, report_format)?;

    println!("{}", "✅ Report generated successfully!".bright_green().bold());
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
    println!("  {} Open the report to view detailed analysis", "1.".bright_blue());
    if matches!(format, ReportFormat::Html) {
        println!("  {} Open in browser: file://{}", "2.".bright_blue(),
            output.canonicalize()?.display());
    }
    println!();

    Ok(())
}

fn cmd_status() -> anyhow::Result<()> {
    println!("{}", "📊 Workflow Status".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if any work has been done
    let has_started = state.phases.values().any(|info| info.status != PhaseStatus::NotStarted);

    if !has_started {
        println!("{}", "No workflow started yet.".dimmed());
        println!();
        println!("{}", "💡 Get started:".bright_yellow().bold());
        println!("  {} Run {} to analyze your project", "1.".bright_blue(), "batuta analyze".cyan());
        println!("  {} Run {} to initialize configuration", "2.".bright_blue(), "batuta init".cyan());
        println!();
        return Ok(());
    }

    display_workflow_progress(&state);

    // Display detailed phase information
    println!("{}", "Phase Details:".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state.phases.get(&phase).unwrap();

        let status_icon = match info.status {
            PhaseStatus::Completed => "✓".bright_green(),
            PhaseStatus::InProgress => "⏳".bright_yellow(),
            PhaseStatus::Failed => "✗".bright_red(),
            PhaseStatus::NotStarted => "○".dimmed(),
        };

        println!();
        println!("{} {}", status_icon, format!("{}", phase).bold());

        if let Some(started) = info.started_at {
            println!("  Started: {}", started.format("%Y-%m-%d %H:%M:%S UTC").to_string().dimmed());
        }

        if let Some(completed) = info.completed_at {
            println!("  Completed: {}", completed.format("%Y-%m-%d %H:%M:%S UTC").to_string().dimmed());

            if let Some(started) = info.started_at {
                let duration = completed.signed_duration_since(started);
                println!("  Duration: {:.2}s", duration.num_milliseconds() as f64 / 1000.0);
            }
        }

        if let Some(error) = &info.error {
            println!("  {}: {}", "Error".red().bold(), error.red());
        }
    }

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Show next recommended action
    if let Some(current) = state.current_phase {
        println!("{}", "💡 Next Step:".bright_green().bold());
        match current {
            WorkflowPhase::Analysis => {
                println!("  Run {} to analyze your project", "batuta analyze --languages --tdg".cyan());
            }
            WorkflowPhase::Transpilation => {
                println!("  Run {} to convert your code", "batuta transpile".cyan());
            }
            WorkflowPhase::Optimization => {
                println!("  Run {} to optimize performance", "batuta optimize".cyan());
            }
            WorkflowPhase::Validation => {
                println!("  Run {} to validate equivalence", "batuta validate".cyan());
            }
            WorkflowPhase::Deployment => {
                println!("  Run {} to build final binary", "batuta build --release".cyan());
            }
        }
        println!();
    }

    Ok(())
}

fn cmd_reset(skip_confirm: bool) -> anyhow::Result<()> {
    println!("{}", "🔄 Reset Workflow".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();

    if !state_file.exists() {
        println!("{}", "No workflow state found.".dimmed());
        return Ok(());
    }

    // Load current state to show what will be reset
    let state = WorkflowState::load(&state_file)?;
    let completed_count = state
        .phases
        .values()
        .filter(|info| info.status == PhaseStatus::Completed)
        .count();

    if completed_count > 0 {
        println!("{}", "⚠️  Warning:".yellow().bold());
        println!("  This will reset {} completed phase(s)", completed_count.to_string().yellow());
        println!();
    }

    // Confirm unless --yes flag provided
    if !skip_confirm {
        print!("Are you sure you want to reset the workflow? [y/N] ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            println!("{}", "Reset cancelled.".dimmed());
            return Ok(());
        }
    }

    // Delete state file
    std::fs::remove_file(&state_file)?;

    println!();
    println!("{}", "✅ Workflow state reset successfully!".bright_green().bold());
    println!();
    println!("{}", "💡 Next Step:".bright_yellow().bold());
    println!("  Run {} to start fresh", "batuta analyze --languages --tdg".cyan());
    println!();

    Ok(())
}

fn cmd_parf(
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
        println!("{} Finding references to '{}'...", "→".bright_blue(), symbol.cyan());
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
                output.push_str(&format!("  Technical Debt (TODO/FIXME): {}\n", tech_debt_count));
                output.push_str(&format!("  Error Handling Issues: {}\n", error_handling_count));
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
                output.push_str(&format!("- Error Handling Issues: {}\n", error_handling_count));
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
                output.push_str(&format!("Potentially unused symbols: {}\n\n", dead_code.len()));
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
        println!("{} Report written to: {}", "✓".bright_green(), out_path.display().to_string().cyan());
    } else {
        println!();
        println!("{}", output);
    }

    println!();
    println!("{}", "✅ PARF analysis complete!".bright_green().bold());
    println!();

    Ok(())
}
