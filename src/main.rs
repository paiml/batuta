// CLI binary is only available for native targets (not WASM)
#![cfg(feature = "native")]

mod analyzer;
mod ansi_colors;
mod backend;
mod bug_hunter;
mod cli;
mod comply;
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

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

        /// Enable RAG profiling output (timing breakdown)
        #[arg(long)]
        rag_profile: bool,

        /// Enable RAG tracing (detailed query execution trace)
        #[arg(long)]
        rag_trace: bool,

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

        /// Search functions via PMAT quality-annotated code search
        #[arg(long)]
        pmat_query: bool,

        /// Project path for PMAT query (defaults to current directory)
        #[arg(long)]
        pmat_project_path: Option<String>,

        /// Maximum number of PMAT results to return
        #[arg(long, default_value = "10")]
        pmat_limit: usize,

        /// Minimum TDG grade filter (A, B, C, D, F)
        #[arg(long)]
        pmat_min_grade: Option<String>,

        /// Maximum cyclomatic complexity filter
        #[arg(long)]
        pmat_max_complexity: Option<u32>,

        /// Include source code in PMAT results
        #[arg(long)]
        pmat_include_source: bool,

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

    /// Proactive Bug Hunting (Popperian Falsification-Driven Defect Discovery)
    ///
    /// Systematically attempt to falsify the implicit claim "this code is correct"
    /// using multiple hunting strategies: mutation testing, SBFL, static analysis,
    /// fuzzing, and concolic execution.
    ///
    /// Examples:
    ///   batuta bug-hunter analyze .           # LLM-augmented static analysis
    ///   batuta bug-hunter hunt --stack-trace crash.log  # SBFL from crash
    ///   batuta bug-hunter falsify --target src/lib.rs   # Mutation testing
    ///   batuta bug-hunter fuzz --target-unsafe          # Fuzz unsafe blocks
    ///   batuta bug-hunter ensemble .          # Run all modes and combine
    #[command(name = "bug-hunter")]
    BugHunter {
        #[command(subcommand)]
        command: cli::bug_hunter::BugHunterCommand,
    },
}

// Use enums from cli::pipeline_cmds for CLI argument parsing
use cli::pipeline_cmds::{OptimizationProfile, ReportFormat};

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

    // Drift check: runs once per session in local dev, always in CI
    // --allow-drift or --unsafe-skip-drift-check skip the check entirely
    if !cli.unsafe_skip_drift_check && !cli.allow_drift {
        // In local dev, this will only show warning once per session
        enforce_drift_check(cli.strict, &cli.command)?;
    }

    dispatch_command(cli.command)
}

/// Get the drift warning marker file path (workspace-scoped).
/// Uses workspace root hash to scope warnings per project.
fn drift_marker_path() -> std::path::PathBuf {
    // Hash the workspace root to scope warnings per project
    let workspace_id = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| {
            // Simple hash: sum of bytes mod 100000
            s.bytes().map(|b| b as u64).sum::<u64>() % 100000
        }))
        .unwrap_or(0);
    std::env::temp_dir().join(format!("batuta-drift-shown-{}", workspace_id))
}

/// Check if drift warning was already shown this session.
fn drift_already_shown() -> bool {
    let marker = drift_marker_path();
    if marker.exists() {
        // Check if marker is less than 1 hour old
        if let Ok(meta) = std::fs::metadata(&marker) {
            if let Ok(modified) = meta.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    return elapsed.as_secs() < 3600; // 1 hour
                }
            }
        }
    }
    false
}

/// Mark drift warning as shown for this session.
fn mark_drift_shown() {
    let _ = std::fs::write(drift_marker_path(), "shown");
}

/// Enforce stack drift checking with smart tolerance.
///
/// In local dev (git workspace): shows warning ONCE per session, never blocks.
/// In CI (strict mode): always blocks on drift.
fn enforce_drift_check(strict: bool, command: &Commands) -> anyhow::Result<()> {
    let strict_mode = strict || is_strict_env();
    let is_local_dev = is_git_workspace();

    // In local dev, only check once per session (Muda elimination)
    if is_local_dev && !strict_mode && drift_already_shown() {
        return Ok(());
    }

    let Some(drifts) = check_stack_drift()? else {
        return Ok(());
    };
    if drifts.is_empty() {
        return Ok(());
    }

    let read_only = is_read_only_command(command);

    if strict_mode {
        eprintln!("{}", stack::format_drift_errors(&drifts));
        std::process::exit(1);
    } else if read_only || is_local_dev {
        // Local dev: warn once, never block
        warn!("Stack drift detected (non-blocking in local dev)");
        eprintln!("{}", format_drift_warning(&drifts));
        mark_drift_shown(); // Don't show again this session
    } else {
        eprintln!("{}", stack::format_drift_errors(&drifts));
        std::process::exit(1);
    }

    Ok(())
}

/// Dispatch CLI command to the appropriate handler.
fn dispatch_command(command: Commands) -> anyhow::Result<()> {
    match command {
        Commands::Init { source, output } => {
            info!("Initializing Batuta project from {:?}", source);
            cli::pipeline_cmds::cmd_init(source, output)
        }
        Commands::Analyze {
            path,
            tdg,
            languages,
            dependencies,
        } => {
            info!("Analyzing project at {:?}", path);
            cli::pipeline_cmds::cmd_analyze(path, tdg, languages, dependencies)
        }
        Commands::Transpile {
            incremental,
            cache,
            modules,
            ruchy,
            repl,
        } => {
            info!("Transpiling to {}", if ruchy { "Ruchy" } else { "Rust" });
            cli::pipeline_cmds::cmd_transpile(incremental, cache, modules, ruchy, repl)
        }
        Commands::Optimize {
            enable_gpu,
            enable_simd,
            profile,
            gpu_threshold,
        } => {
            info!("Optimizing with profile: {:?}", profile);
            cli::pipeline_cmds::cmd_optimize(enable_gpu, enable_simd, profile, gpu_threshold)
        }
        Commands::Validate {
            trace_syscalls,
            diff_output,
            run_original_tests,
            benchmark,
        } => {
            info!("Validating semantic equivalence");
            cli::pipeline_cmds::cmd_validate(trace_syscalls, diff_output, run_original_tests, benchmark)
        }
        Commands::Build {
            release,
            target,
            wasm,
        } => {
            info!("Building Rust project");
            cli::pipeline_cmds::cmd_build(release, target, wasm)
        }
        Commands::Report { output, format } => {
            info!("Generating migration report");
            cli::pipeline_cmds::cmd_report(output, format)
        }
        Commands::Status => {
            info!("Checking workflow status");
            cli::workflow::cmd_status()
        }
        Commands::Reset { yes } => {
            info!("Resetting workflow state");
            cli::workflow::cmd_reset(yes)
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
            )
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
            rag_profile,
            rag_trace,
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
            pmat_query,
            pmat_project_path,
            pmat_limit,
            pmat_min_grade,
            pmat_max_complexity,
            pmat_include_source,
            format,
        } => dispatch_oracle(
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
            rag_profile,
            rag_trace,
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
            pmat_query,
            pmat_project_path,
            pmat_limit,
            pmat_min_grade,
            pmat_max_complexity,
            pmat_include_source,
            format,
        ),
        Commands::Stack { command } => {
            info!("Stack Mode");
            cli::stack::cmd_stack(command)
        }
        Commands::Hf { command } => {
            info!("HuggingFace Mode");
            cli::hf::cmd_hf(command)
        }
        Commands::Pacha { command } => {
            info!("Pacha Model Registry Mode");
            pacha::cmd_pacha(command)
        }
        Commands::Data { command } => {
            info!("Data Platforms Mode");
            cli::data::cmd_data(command)
        }
        Commands::Viz { command } => {
            info!("Visualization Frameworks Mode");
            cli::viz::cmd_viz(command)
        }
        Commands::Experiment { command } => {
            info!("Experiment Tracking Frameworks Mode");
            cli::experiment::cmd_experiment(command)
        }
        Commands::Content { command } => {
            info!("Content Creation Tooling Mode");
            cli::content::cmd_content(command)
        }
        Commands::Serve {
            model,
            host,
            port,
            openai_api,
            watch,
        } => {
            info!("Starting Model Server Mode");
            cli::serve::cmd_serve(model, &host, port, openai_api, watch)
        }
        Commands::Deploy { command } => {
            info!("Deployment Generation Mode");
            cli::deploy::cmd_deploy(command)
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
            cli::falsify::cmd_falsify(path, critical_only, format, output, &min_grade, verbose)
        }
        Commands::BugHunter { command } => {
            info!("Proactive Bug Hunting Mode");
            cli::bug_hunter::handle_bug_hunter_command(command)
                .map_err(|e| anyhow::anyhow!(e))
        }
    }
}

/// Try dispatching an Oracle RAG subcommand.
#[allow(clippy::too_many_arguments)]
fn try_oracle_rag(
    query: &Option<String>,
    rag: bool,
    rag_index: bool,
    rag_index_force: bool,
    rag_stats: bool,
    rag_profile: bool,
    rag_trace: bool,
    #[cfg(feature = "native")] rag_dashboard: bool,
    format: cli::oracle::OracleOutputFormat,
) -> Option<anyhow::Result<()>> {
    #[cfg(feature = "native")]
    if rag_dashboard {
        return Some(cli::oracle::cmd_oracle_rag_dashboard());
    }
    if rag_stats {
        return Some(cli::oracle::cmd_oracle_rag_stats(format));
    }
    if rag_index || rag_index_force {
        return Some(cli::oracle::cmd_oracle_rag_index(rag_index_force));
    }
    if rag {
        return Some(cli::oracle::cmd_oracle_rag_with_profile(
            query.clone(),
            format,
            rag_profile,
            rag_trace,
        ));
    }
    None
}

/// Try dispatching a specialized Oracle subcommand (local/RAG/pmat-query/cookbook).
/// Returns `Some(result)` if a subcommand matched, `None` for default classic oracle.
#[allow(clippy::too_many_arguments)]
fn try_oracle_subcommand(
    query: &Option<String>,
    local: bool,
    dirty: bool,
    publish_order: bool,
    rag: bool,
    rag_index: bool,
    rag_index_force: bool,
    rag_stats: bool,
    rag_profile: bool,
    rag_trace: bool,
    #[cfg(feature = "native")] rag_dashboard: bool,
    pmat_query: bool,
    pmat_project_path: &Option<String>,
    pmat_limit: usize,
    pmat_min_grade: &Option<String>,
    pmat_max_complexity: Option<u32>,
    pmat_include_source: bool,
    cookbook: bool,
    recipe: &Option<String>,
    recipes_by_tag: &Option<String>,
    recipes_by_component: &Option<String>,
    search_recipes: &Option<String>,
    format: cli::oracle::OracleOutputFormat,
) -> Option<anyhow::Result<()>> {
    if local || dirty || publish_order {
        return Some(cli::oracle::cmd_oracle_local(
            local,
            dirty,
            publish_order,
            format,
        ));
    }

    let rag_result = try_oracle_rag(
        query,
        rag,
        rag_index,
        rag_index_force,
        rag_stats,
        rag_profile,
        rag_trace,
        #[cfg(feature = "native")]
        rag_dashboard,
        format,
    );
    if rag_result.is_some() {
        return rag_result;
    }

    if pmat_query {
        return Some(cli::oracle::cmd_oracle_pmat_query(
            query.clone(),
            pmat_project_path.clone(),
            pmat_limit,
            pmat_min_grade.clone(),
            pmat_max_complexity,
            pmat_include_source,
            rag,
            format,
        ));
    }

    if cookbook
        || recipe.is_some()
        || recipes_by_tag.is_some()
        || recipes_by_component.is_some()
        || search_recipes.is_some()
    {
        return Some(cli::oracle::cmd_oracle_cookbook(
            cookbook,
            recipe.clone(),
            recipes_by_tag.clone(),
            recipes_by_component.clone(),
            search_recipes.clone(),
            format,
        ));
    }
    None
}

/// Handle Oracle subcommand dispatch (many boolean/option flags).
#[allow(clippy::too_many_arguments)]
fn dispatch_oracle(
    query: Option<String>,
    recommend: bool,
    problem: Option<String>,
    data_size: Option<String>,
    integrate: Option<String>,
    capabilities: Option<String>,
    list: bool,
    show: Option<String>,
    interactive: bool,
    rag: bool,
    rag_index: bool,
    rag_index_force: bool,
    rag_stats: bool,
    rag_profile: bool,
    rag_trace: bool,
    #[cfg(feature = "native")] rag_dashboard: bool,
    cookbook: bool,
    recipe: Option<String>,
    recipes_by_tag: Option<String>,
    recipes_by_component: Option<String>,
    search_recipes: Option<String>,
    local: bool,
    dirty: bool,
    publish_order: bool,
    pmat_query: bool,
    pmat_project_path: Option<String>,
    pmat_limit: usize,
    pmat_min_grade: Option<String>,
    pmat_max_complexity: Option<u32>,
    pmat_include_source: bool,
    format: cli::oracle::OracleOutputFormat,
) -> anyhow::Result<()> {
    info!("Oracle Mode");

    if let Some(result) = try_oracle_subcommand(
        &query,
        local,
        dirty,
        publish_order,
        rag,
        rag_index,
        rag_index_force,
        rag_stats,
        rag_profile,
        rag_trace,
        #[cfg(feature = "native")]
        rag_dashboard,
        pmat_query,
        &pmat_project_path,
        pmat_limit,
        &pmat_min_grade,
        pmat_max_complexity,
        pmat_include_source,
        cookbook,
        &recipe,
        &recipes_by_tag,
        &recipes_by_component,
        &search_recipes,
        format,
    ) {
        return result;
    }

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
    })
}
