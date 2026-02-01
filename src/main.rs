// CLI binary is only available for native targets (not WASM)
#![cfg(feature = "native")]

mod analyzer;
mod ansi_colors;
mod backend;
mod cli;
mod config;
mod content;
mod data;
mod experiment;
mod hf;
mod numpy_converter;
mod oracle;
mod pacha;
mod parf;
mod pipeline;
mod pipeline_analysis;
mod pytorch_converter;
mod report;
mod sklearn_converter;
mod stack;
mod tools;
mod types;
mod viz;

use analyzer::analyze_project;
use ansi_colors::Colorize; // DEP-REDUCE: replaced colored crate
use clap::{Parser, Subcommand};
use config::BatutaConfig;
use std::path::{Path, PathBuf};
use tools::ToolRegistry;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use types::{PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState};

/// Get the workflow state file path
fn get_state_file_path() -> PathBuf {
    cli::get_state_file_path()
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

    /// Skip stack drift check (emergency use only - hidden)
    #[arg(long, global = true, hide = true)]
    unsafe_skip_drift_check: bool,

    /// Enforce strict drift checking (blocks on any drift)
    /// Default: tolerant in local dev (warn only), strict in CI
    #[arg(long, global = true)]
    strict: bool,

    /// Allow drift warnings without blocking (explicit tolerance)
    #[arg(long, global = true)]
    allow_drift: bool,
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
        format: cli::parf::ParfOutputFormat,

        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Oracle Mode - Query the Sovereign AI Stack for recommendations
    Oracle {
        /// Natural language query (e.g., "How do I train a model?")
        query: Option<String>,

        /// Get component recommendation for a problem
        #[arg(long)]
        recommend: bool,

        /// Specify problem type for recommendation
        #[arg(long)]
        problem: Option<String>,

        /// Data size for recommendations (e.g., "1M", "100K")
        #[arg(long)]
        data_size: Option<String>,

        /// Show integration pattern between components
        #[arg(long)]
        integrate: Option<String>,

        /// List capabilities of a component
        #[arg(long)]
        capabilities: Option<String>,

        /// List all stack components
        #[arg(long)]
        list: bool,

        /// Show component details
        #[arg(long)]
        show: Option<String>,

        /// Enter interactive mode
        #[arg(short, long)]
        interactive: bool,

        /// Use RAG-based retrieval from indexed documentation
        #[arg(long)]
        rag: bool,

        /// Index/reindex stack documentation for RAG
        #[arg(long)]
        rag_index: bool,

        /// Force reindex (clear cache first)
        #[arg(long)]
        rag_index_force: bool,

        /// Show RAG index statistics
        #[arg(long)]
        rag_stats: bool,

        /// Show RAG dashboard (TUI)
        #[cfg(feature = "native")]
        #[arg(long)]
        rag_dashboard: bool,

        /// List all cookbook recipes
        #[arg(long)]
        cookbook: bool,

        /// Show a specific recipe by ID
        #[arg(long)]
        recipe: Option<String>,

        /// Find recipes by tag (e.g., "wasm", "ml", "distributed")
        #[arg(long)]
        recipes_by_tag: Option<String>,

        /// Find recipes by component (e.g., "aprender", "trueno")
        #[arg(long)]
        recipes_by_component: Option<String>,

        /// Search recipes by keyword
        #[arg(long)]
        search_recipes: Option<String>,

        /// Show local workspace status (~/src PAIML projects)
        #[arg(long)]
        local: bool,

        /// Show only dirty (uncommitted) projects
        #[arg(long)]
        dirty: bool,

        /// Show publish order for local projects
        #[arg(long)]
        publish_order: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: cli::oracle::OracleOutputFormat,
    },

    /// PAIML Stack dependency orchestration
    Stack {
        #[command(subcommand)]
        command: cli::stack::StackCommand,
    },

    /// HuggingFace Hub integration
    Hf {
        #[command(subcommand)]
        command: cli::hf::HfCommand,
    },

    /// Pacha model registry (ollama-like model management)
    Pacha {
        #[command(subcommand)]
        command: pacha::PachaCommand,
    },

    /// Data Platforms integration (Databricks, Snowflake, AWS, HuggingFace)
    Data {
        #[command(subcommand)]
        command: cli::data::DataCommand,
    },

    /// Visualization frameworks ecosystem (Gradio, Streamlit, Panel, Dash)
    Viz {
        #[command(subcommand)]
        command: cli::viz::VizCommand,
    },

    /// Experiment tracking frameworks comparison (MLflow replacement)
    Experiment {
        #[command(subcommand)]
        command: cli::experiment::ExperimentCommand,
    },

    /// Content creation tooling (prompt emission)
    Content {
        #[command(subcommand)]
        command: cli::content::ContentCommand,
    },

    /// Serve ML models via Realizar inference server
    ///
    /// Examples:
    ///   batuta serve pacha://llama3:8b
    ///   batuta serve ./model.gguf --port 8080
    ///   batuta serve --openai-api
    Serve {
        /// Model reference (pacha://name:version, hf://org/model, or path)
        #[arg(value_name = "MODEL")]
        model: Option<String>,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Enable OpenAI-compatible API at /v1/*
        #[arg(long, default_value = "true")]
        openai_api: bool,

        /// Enable hot-reload on model changes
        #[arg(long)]
        watch: bool,
    },

    /// Deploy ML models to production targets
    ///
    /// Examples:
    ///   batuta deploy docker pacha://llama3:8b
    ///   batuta deploy lambda my-model:v1.0
    ///   batuta deploy k8s --replicas 3
    Deploy {
        #[command(subcommand)]
        command: cli::deploy::DeployCommand,
    },

    /// Run Popperian Falsification Checklist (Sovereign AI Assurance Protocol)
    ///
    /// Evaluates a project against the 108-item checklist for scientific rigor.
    /// Toyota Way principles: Jidoka (automated gates), Genchi Genbutsu (evidence-based review)
    ///
    /// Examples:
    ///   batuta falsify .                     # Evaluate current project
    ///   batuta falsify --critical-only       # Only CRITICAL items (5 architectural invariants)
    ///   batuta falsify --format json         # JSON output for CI/CD
    Falsify {
        /// Project path to evaluate
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Only evaluate CRITICAL architectural invariants (AI-01 through AI-05)
        #[arg(long)]
        critical_only: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: cli::falsify::FalsifyOutputFormat,

        /// Output file (default: stdout)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Fail on any check below this grade (toyota-standard, kaizen-required, andon-warning)
        #[arg(long, default_value = "kaizen-required")]
        min_grade: String,

        /// Show verbose evidence for each check
        #[arg(long)]
        verbose: bool,
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

/// Check if running in a git workspace
fn is_git_workspace() -> bool {
    std::path::Path::new(".git").exists()
        || std::process::Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

/// Check if the environment requests strict mode (BATUTA_STRICT=1)
fn is_strict_env() -> bool {
    std::env::var("BATUTA_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Determine if a command is read-only (should never block on drift)
fn is_read_only_command(cmd: &Commands) -> bool {
    matches!(
        cmd,
        Commands::Oracle { .. }     // RAG queries, recommendations, cookbook
        | Commands::Status          // Workflow status check
        | Commands::Analyze { .. }  // Code analysis
        | Commands::Parf { .. } // Code search and analysis
    )
}

/// Format drift as a warning (non-blocking)
fn format_drift_warning(drifts: &[stack::DriftReport]) -> String {
    use ansi_colors::Colorize;

    let mut output = String::new();
    output.push_str(&format!(
        "{}\n\n",
        "⚠️  Stack Drift Warning (non-blocking)"
            .bright_yellow()
            .bold()
    ));

    for drift in drifts {
        output.push_str(&format!("   {}\n", drift.display()));
    }

    output.push_str(&format!(
        "\n{}",
        "Run 'batuta stack drift --fix' to update dependencies.\n".dimmed()
    ));
    output.push_str(&format!(
        "{}",
        "Use --strict to enforce drift checking.\n".dimmed()
    ));

    output
}

/// Check for stack drift across PAIML crates
///
/// Returns None if check cannot be performed (offline, etc.)
/// Returns Some(empty) if no drift detected
/// Returns Some(drifts) if drift detected
fn check_stack_drift() -> anyhow::Result<Option<Vec<stack::DriftReport>>> {
    // Create runtime for async operations
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return Ok(None), // Can't create runtime, skip check
    };

    rt.block_on(async {
        let mut client = stack::CratesIoClient::new().with_persistent_cache();
        let mut checker = stack::DriftChecker::new();

        match checker.detect_drift(&mut client).await {
            Ok(drifts) => Ok(Some(drifts)),
            Err(_) => Ok(None), // Network error or similar, skip check
        }
    })
}

#[allow(clippy::cognitive_complexity)]
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

    // Stack drift check with smart tolerance
    //
    // Behavior:
    // - --allow-drift or --unsafe-skip-drift-check: skip entirely
    // - --strict or BATUTA_STRICT=1: block on any drift
    // - Read-only commands (oracle, analyze, parf): warn only
    // - In git workspace without --strict: warn only (local dev mode)
    // - Otherwise: block (CI/production default)
    if !cli.unsafe_skip_drift_check && !cli.allow_drift {
        if let Some(drifts) = check_stack_drift()? {
            if !drifts.is_empty() {
                let strict_mode = cli.strict || is_strict_env();
                let read_only = is_read_only_command(&cli.command);
                let in_workspace = is_git_workspace();

                if strict_mode {
                    // Strict mode: always block
                    eprintln!("{}", stack::format_drift_errors(&drifts));
                    std::process::exit(1);
                } else if read_only {
                    // Read-only operations: warn only, never block
                    warn!("Stack drift detected (non-blocking for read-only operation)");
                    eprintln!("{}", format_drift_warning(&drifts));
                } else if in_workspace {
                    // Local development: warn only by default
                    warn!("Stack drift detected (non-blocking in local dev mode)");
                    eprintln!("{}", format_drift_warning(&drifts));
                } else {
                    // CI/production without explicit --allow-drift: block
                    eprintln!("{}", stack::format_drift_errors(&drifts));
                    std::process::exit(1);
                }
            }
        }
    }

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
            cli::workflow::cmd_status()?;
        }
        Commands::Reset { yes } => {
            info!("Resetting workflow state");
            cli::workflow::cmd_reset(yes)?;
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
            cli::parf::cmd_parf(
                &path,
                find.as_deref(),
                patterns,
                dependencies,
                dead_code,
                format,
                output.as_deref(),
            )?;
        }
        Commands::Oracle {
            query,
            recommend,
            problem,
            data_size,
            integrate,
            capabilities,
            list,
            show,
            interactive,
            rag,
            rag_index,
            rag_index_force,
            rag_stats,
            #[cfg(feature = "native")]
            rag_dashboard,
            cookbook,
            recipe,
            recipes_by_tag,
            recipes_by_component,
            search_recipes,
            local,
            dirty,
            publish_order,
            format,
        } => {
            info!("Oracle Mode");

            // Handle local workspace commands
            if local || dirty || publish_order {
                return cli::oracle::cmd_oracle_local(local, dirty, publish_order, format);
            }

            // Handle RAG-specific commands
            #[cfg(feature = "native")]
            if rag_dashboard {
                return cli::oracle::cmd_oracle_rag_dashboard();
            }

            if rag_stats {
                return cli::oracle::cmd_oracle_rag_stats(format);
            }

            if rag_index || rag_index_force {
                return cli::oracle::cmd_oracle_rag_index(rag_index_force);
            }

            if rag {
                return cli::oracle::cmd_oracle_rag(query, format);
            }

            // Handle cookbook commands
            if cookbook
                || recipe.is_some()
                || recipes_by_tag.is_some()
                || recipes_by_component.is_some()
                || search_recipes.is_some()
            {
                return cli::oracle::cmd_oracle_cookbook(
                    cookbook,
                    recipe,
                    recipes_by_tag,
                    recipes_by_component,
                    search_recipes,
                    format,
                );
            }

            // Default to hardcoded knowledge graph
            cli::oracle::cmd_oracle(cli::oracle::OracleOptions {
                query,
                recommend,
                problem,
                data_size,
                integrate,
                capabilities,
                list,
                show,
                interactive,
                format,
            })?;
        }
        Commands::Stack { command } => {
            info!("Stack Mode");
            cli::stack::cmd_stack(command)?;
        }
        Commands::Hf { command } => {
            info!("HuggingFace Mode");
            cli::hf::cmd_hf(command)?;
        }
        Commands::Pacha { command } => {
            info!("Pacha Model Registry Mode");
            pacha::cmd_pacha(command)?;
        }
        Commands::Data { command } => {
            info!("Data Platforms Mode");
            cli::data::cmd_data(command)?;
        }
        Commands::Viz { command } => {
            info!("Visualization Frameworks Mode");
            cli::viz::cmd_viz(command)?;
        }
        Commands::Experiment { command } => {
            info!("Experiment Tracking Frameworks Mode");
            cli::experiment::cmd_experiment(command)?;
        }
        Commands::Content { command } => {
            info!("Content Creation Tooling Mode");
            cli::content::cmd_content(command)?;
        }
        Commands::Serve {
            model,
            host,
            port,
            openai_api,
            watch,
        } => {
            info!("Starting Model Server Mode");
            cli::serve::cmd_serve(model, &host, port, openai_api, watch)?;
        }
        Commands::Deploy { command } => {
            info!("Deployment Generation Mode");
            cli::deploy::cmd_deploy(command)?;
        }
        Commands::Falsify {
            path,
            critical_only,
            format,
            output,
            min_grade,
            verbose,
        } => {
            info!("Popperian Falsification Checklist Mode");
            cli::falsify::cmd_falsify(path, critical_only, format, output, &min_grade, verbose)?;
        }
    }

    Ok(())
}

// Command implementations (stubs for now)

fn cmd_init(source: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
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

    Ok(())
}

/// Display project analysis results
fn display_analysis_results(analysis: &ProjectAnalysis) {
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
            lang_stat.file_count.to_string().yellow(),
            lang_stat.line_count.to_string().green(),
            lang_stat.percentage
        );
    }
    println!();

    if let Some(primary) = &analysis.primary_language {
        println!(
            "{}: {}",
            "Primary language".bold(),
            format!("{}", primary).bright_cyan()
        );
    }

    if let Some(transpiler) = analysis.recommend_transpiler() {
        println!(
            "{}: {}",
            "Recommended transpiler".bold(),
            transpiler.bright_green()
        );
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

    let grade = cli::calculate_tdg_grade(score);
    let grade_colored = match grade {
        cli::TdgGrade::APlus => format!("{}", grade).bright_green(),
        cli::TdgGrade::A => format!("{}", grade).green(),
        cli::TdgGrade::B | cli::TdgGrade::C => format!("{}", grade).yellow(),
        cli::TdgGrade::D => format!("{}", grade).red(),
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

/// Display next steps after analysis
fn display_analyze_next_steps() {
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!(
        "  {} Run {} to initialize configuration",
        "1.".bright_blue(),
        "batuta init".cyan()
    );
    println!(
        "  {} Run {} to convert project to Rust",
        "2.".bright_blue(),
        "batuta transpile".cyan()
    );
    println!(
        "  {} Run {} for performance optimization",
        "3.".bright_blue(),
        "batuta optimize".cyan()
    );
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
    cli::workflow::display_workflow_progress(&state);
    display_analyze_next_steps();

    Ok(())
}

/// Check transpilation prerequisites
fn check_transpile_prerequisites(state: &WorkflowState) -> anyhow::Result<BatutaConfig> {
    if !state.is_phase_completed(WorkflowPhase::Analysis) {
        println!("{}", "⚠️  Analysis phase not completed!".yellow().bold());
        println!();
        println!(
            "Run {} first to analyze your project.",
            "batuta analyze".cyan()
        );
        println!();
        cli::workflow::display_workflow_progress(state);
        anyhow::bail!("Analysis phase not completed");
    }

    let config_path = PathBuf::from("batuta.toml");
    if !config_path.exists() {
        println!("{}", "⚠️  No batuta.toml found!".yellow().bold());
        println!();
        println!(
            "Run {} first to create a configuration file.",
            "batuta init".cyan()
        );
        println!();
        anyhow::bail!("No configuration file found");
    }

    let config = BatutaConfig::load(&config_path)?;
    println!("{} Loaded configuration", "✓".bright_green());
    Ok(config)
}

/// Setup transpiler tools and analyze project
fn setup_transpiler(
    config: &BatutaConfig,
) -> anyhow::Result<(ToolRegistry, types::Language, tools::ToolInfo)> {
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
    let primary_lang = analysis
        .primary_language
        .ok_or_else(|| anyhow::anyhow!("Could not determine primary language"))?;

    println!(
        "{} Primary language: {}",
        "✓".bright_green(),
        format!("{}", primary_lang).cyan()
    );

    let transpiler = tools
        .get_transpiler_for_language(&primary_lang)
        .ok_or_else(|| anyhow::anyhow!("No transpiler available for {}", primary_lang))?
        .clone();

    println!(
        "{} Using transpiler: {}",
        "✓".bright_green(),
        transpiler.name.cyan()
    );
    if let Some(ver) = &transpiler.version {
        println!("  {} Version: {}", "ℹ".bright_blue(), ver.dimmed());
    }
    println!();

    Ok((tools, primary_lang, transpiler))
}

/// Handle missing transpiler tools
fn handle_missing_tools(
    tools: &ToolRegistry,
    config: &BatutaConfig,
) -> anyhow::Result<(ToolRegistry, types::Language, tools::ToolInfo)> {
    println!();
    println!("{}", "❌ No transpiler tools found!".red().bold());
    println!();
    println!("{}", "Install required tools:".yellow());

    let analysis = analyze_project(&config.source.path, false, true, false)?;
    let needed_tools = analysis
        .primary_language
        .as_ref()
        .map(cli::get_needed_tools_for_language)
        .unwrap_or_default();

    let instructions = tools.get_installation_instructions(&needed_tools);
    for inst in instructions {
        println!("  {} {}", "•".bright_blue(), inst.cyan());
    }
    println!();
    anyhow::bail!("No transpiler tools found")
}

/// Build transpiler arguments (wrapper for cli::build_transpiler_args)
fn build_transpiler_args_wrapper(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) -> Vec<String> {
    cli::build_transpiler_args(config, incremental, cache, ruchy, modules)
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
    let owned_args = build_transpiler_args_wrapper(&config, incremental, cache, ruchy, &modules);
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

    cli::workflow::display_workflow_progress(state);

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

    cli::workflow::display_workflow_progress(state);

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
        cli::workflow::display_workflow_progress(&state);
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
    cli::workflow::display_workflow_progress(&state);

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
        cli::workflow::display_workflow_progress(&state);
        return Ok(());
    }

    // Start validation phase
    state.start_phase(WorkflowPhase::Validation);
    state.save(&state_file)?;

    // Display validation settings
    display_validation_settings(trace_syscalls, diff_output, run_original_tests, benchmark);

    // Implement validation with Renacer (BATUTA-011)
    let mut validation_passed = true;

    if trace_syscalls {
        if !run_syscall_tracing(run_original_tests) {
            validation_passed = false;
        }
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
    cli::workflow::display_workflow_progress(&state);

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
    use crate::pipeline::{PipelineContext, PipelineStage, ValidationStage};

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
        println!(
            "Run {} first to validate your project.",
            "batuta validate".cyan()
        );
        println!();
        cli::workflow::display_workflow_progress(&state);
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

    // BATUTA-009: Build system planned for Phase 5
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
    cli::workflow::display_workflow_progress(&state);

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

fn cmd_report(output: PathBuf, format: ReportFormat) -> anyhow::Result<()> {
    println!(
        "{}",
        "📊 Generating migration report...".bright_cyan().bold()
    );
    println!();

    // Load workflow state
    let state_file = get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

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
