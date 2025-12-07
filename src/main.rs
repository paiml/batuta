// CLI binary is only available for native targets (not WASM)
#![cfg(feature = "native")]

mod analyzer;
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
mod pytorch_converter;
mod report;
mod sklearn_converter;
mod stack;
mod tools;
mod types;
mod viz;

use analyzer::analyze_project;
use clap::{Parser, Subcommand};
use colored::Colorize;
use config::BatutaConfig;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tools::ToolRegistry;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use types::{PhaseStatus, ProjectAnalysis, WorkflowPhase, WorkflowState};

/// Get the workflow state file path
fn get_state_file_path() -> PathBuf {
    cli::get_state_file_path()
}

/// Display workflow progress
fn display_workflow_progress(state: &WorkflowState) {
    println!();
    println!("{}", "📊 Workflow Progress".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state
            .phases
            .get(&phase)
            .expect("workflow phase missing from state");
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
            println!(
                "  {} {} [{}]",
                status_icon,
                phase_name.dimmed(),
                status_text.dimmed()
            );
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

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OracleOutputFormat,
    },

    /// PAIML Stack dependency orchestration
    Stack {
        #[command(subcommand)]
        command: StackCommand,
    },

    /// HuggingFace Hub integration
    Hf {
        #[command(subcommand)]
        command: HfCommand,
    },

    /// Pacha model registry (ollama-like model management)
    Pacha {
        #[command(subcommand)]
        command: pacha::PachaCommand,
    },

    /// Data Platforms integration (Databricks, Snowflake, AWS, HuggingFace)
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },

    /// Visualization frameworks ecosystem (Gradio, Streamlit, Panel, Dash)
    Viz {
        #[command(subcommand)]
        command: VizCommand,
    },

    /// Experiment tracking frameworks comparison (MLflow replacement)
    Experiment {
        #[command(subcommand)]
        command: ExperimentCommand,
    },

    /// Content creation tooling (prompt emission)
    Content {
        #[command(subcommand)]
        command: ContentCommand,
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
        command: DeployCommand,
    },
}

/// Stack subcommands for dependency orchestration
#[derive(Subcommand)]
enum StackCommand {
    /// Check dependency health across the PAIML stack
    Check {
        /// Specific project to check (default: all)
        #[arg(long)]
        project: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Fail on any warnings
        #[arg(long)]
        strict: bool,

        /// Verify published crates.io versions exist
        #[arg(long)]
        verify_published: bool,

        /// Skip network requests (use cached crates.io data)
        #[arg(long)]
        offline: bool,

        /// Path to workspace root (default: auto-detect)
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Coordinate releases across the PAIML stack
    Release {
        /// Specific crate to release (releases dependencies first)
        crate_name: Option<String>,

        /// Release all crates with changes
        #[arg(long)]
        all: bool,

        /// Dry run - show what would be released
        #[arg(long)]
        dry_run: bool,

        /// Version bump type for unreleased crates
        #[arg(long, value_enum)]
        bump: Option<BumpType>,

        /// Skip quality gate verification
        #[arg(long)]
        no_verify: bool,

        /// Skip interactive confirmation
        #[arg(long)]
        yes: bool,

        /// Publish to crates.io (default: false for safety)
        #[arg(long)]
        publish: bool,
    },

    /// Show stack health status dashboard
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

    /// Synchronize dependencies across the stack
    Sync {
        /// Specific crate to sync
        crate_name: Option<String>,

        /// Sync all crates
        #[arg(long)]
        all: bool,

        /// Dry run - show what would change
        #[arg(long)]
        dry_run: bool,

        /// Align specific dependency version (e.g., arrow=54.0)
        #[arg(long)]
        align: Option<String>,
    },

    /// Display hierarchical tree of PAIML stack components
    Tree {
        /// Output format (ascii, json, dot)
        #[arg(long, default_value = "ascii")]
        format: String,

        /// Show health status and versions
        #[arg(long)]
        health: bool,

        /// Filter by layer (core, ml, inference, orchestration, distributed, transpilation, docs)
        #[arg(long)]
        filter: Option<String>,
    },

    /// Check quality matrix for PAIML stack components (A+ enforcement)
    Quality {
        /// Specific component to check
        #[arg(long)]
        component: Option<String>,

        /// Require A+ for all components (strict mode)
        #[arg(long)]
        strict: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,

        /// Verify hero images exist and are valid
        #[arg(long)]
        verify_hero: bool,

        /// Show detailed score breakdown
        #[arg(long, short)]
        verbose: bool,

        /// Path to workspace root (default: current directory)
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Quality gate enforcement for CI/pre-commit hooks
    ///
    /// Fails with non-zero exit code if any downstream component is below A- threshold.
    /// Use in pre-commit hooks or CI pipelines to prevent commits/deployments when
    /// quality standards are not met.
    Gate {
        /// Path to workspace root (default: parent of current directory)
        #[arg(long)]
        workspace: Option<PathBuf>,

        /// Quiet mode - only output on failure
        #[arg(long, short)]
        quiet: bool,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum StackOutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum BumpType {
    /// Increment patch version (0.0.x)
    Patch,
    /// Increment minor version (0.x.0)
    Minor,
    /// Increment major version (x.0.0)
    Major,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum OracleOutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
}

/// HuggingFace subcommands
#[derive(Subcommand)]
enum HfCommand {
    /// Display HuggingFace ecosystem tree
    Tree {
        /// Show PAIML-HuggingFace integration map
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },

    /// Search HuggingFace Hub (models, datasets, spaces)
    Search {
        /// Asset type to search
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Search query
        query: String,

        /// Filter by task (for models)
        #[arg(long)]
        task: Option<String>,

        /// Limit results
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// Get info about a HuggingFace asset
    Info {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        repo_id: String,
    },

    /// Pull model/dataset from HuggingFace Hub
    Pull {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Repository ID
        repo_id: String,

        /// Output directory
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Model quantization (for models)
        #[arg(long)]
        quantization: Option<String>,
    },

    /// Push model/dataset to HuggingFace Hub
    Push {
        /// Asset type
        #[arg(value_enum)]
        asset_type: HfAssetType,

        /// Local path to push
        path: PathBuf,

        /// Target repository ID
        #[arg(long)]
        repo: String,

        /// Commit message
        #[arg(long, default_value = "Update from batuta")]
        message: String,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum HfAssetType {
    /// Model repository
    Model,
    /// Dataset repository
    Dataset,
    /// Space (app) repository
    Space,
}

/// Data Platforms subcommands
#[derive(Subcommand)]
enum DataCommand {
    /// Display data platforms ecosystem tree
    Tree {
        /// Filter by platform (databricks, snowflake, aws, huggingface)
        #[arg(long)]
        platform: Option<String>,

        /// Show PAIML integration mappings
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },
}

/// Visualization Frameworks subcommands
#[derive(Subcommand)]
enum VizCommand {
    /// Display visualization frameworks ecosystem tree
    Tree {
        /// Filter by framework (gradio, streamlit, panel, dash)
        #[arg(long)]
        framework: Option<String>,

        /// Show PAIML replacement mappings
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },

    /// Launch monitoring dashboard with Presentar
    Dashboard {
        /// Data source (trueno-db://metrics, prometheus://localhost:9090)
        #[arg(long, default_value = "trueno-db://metrics")]
        source: String,

        /// Dashboard port
        #[arg(long, default_value = "3000")]
        port: u16,

        /// Dashboard theme (light, dark)
        #[arg(long, default_value = "dark")]
        theme: String,

        /// Output config file instead of launching
        #[arg(long)]
        output: Option<String>,
    },
}

/// Experiment tracking subcommands
#[derive(Subcommand)]
enum ExperimentCommand {
    /// Display experiment tracking frameworks ecosystem tree
    Tree {
        /// Filter by framework (mlflow, wandb, neptune, dvc)
        #[arg(long)]
        framework: Option<String>,

        /// Show PAIML replacement mappings
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },
}

/// Content creation subcommands
#[derive(Subcommand)]
enum ContentCommand {
    /// Emit a prompt for content generation
    Emit {
        /// Content type (hlo, dlo, bch, blp, pdm)
        #[arg(long, short = 't')]
        r#type: String,

        /// Title or topic
        #[arg(long)]
        title: Option<String>,

        /// Target audience
        #[arg(long)]
        audience: Option<String>,

        /// Target word count
        #[arg(long)]
        word_count: Option<usize>,

        /// Source context paths (comma-separated)
        #[arg(long)]
        source_context: Option<String>,

        /// Show token budget breakdown
        #[arg(long)]
        show_budget: bool,

        /// Output file (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<std::path::PathBuf>,

        /// Course level for detailed outlines (short, standard, extended)
        #[arg(long, short = 'l')]
        level: Option<String>,
    },

    /// Validate generated content
    Validate {
        /// Content type (hlo, dlo, bch, blp, pdm)
        #[arg(long, short = 't')]
        r#type: String,

        /// File to validate
        file: std::path::PathBuf,

        /// Use LLM-as-a-Judge for style validation
        #[arg(long)]
        llm_judge: bool,
    },

    /// List available content types
    Types,
}

/// Deploy subcommands for production deployment
#[derive(Subcommand)]
enum DeployCommand {
    /// Generate Docker deployment
    Docker {
        /// Model reference
        model: String,

        /// Output directory for Dockerfile
        #[arg(long, short = 'o', default_value = ".")]
        output: PathBuf,

        /// Base image
        #[arg(long, default_value = "rust:slim")]
        base_image: String,

        /// Expose port
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Build multi-stage (smaller image)
        #[arg(long, default_value = "true")]
        multi_stage: bool,
    },

    /// Generate AWS Lambda deployment
    Lambda {
        /// Model reference
        model: String,

        /// Output directory for Lambda package
        #[arg(long, short = 'o', default_value = ".")]
        output: PathBuf,

        /// Memory size (MB)
        #[arg(long, default_value = "1024")]
        memory: u32,

        /// Timeout (seconds)
        #[arg(long, default_value = "30")]
        timeout: u32,

        /// Create SAM template
        #[arg(long)]
        sam: bool,
    },

    /// Generate Kubernetes deployment
    K8s {
        /// Model reference
        model: String,

        /// Output directory for manifests
        #[arg(long, short = 'o', default_value = ".")]
        output: PathBuf,

        /// Number of replicas
        #[arg(long, default_value = "1")]
        replicas: u32,

        /// Namespace
        #[arg(long, default_value = "default")]
        namespace: String,

        /// Create Helm chart
        #[arg(long)]
        helm: bool,

        /// Enable autoscaling
        #[arg(long)]
        hpa: bool,
    },

    /// Generate Fly.io deployment
    Fly {
        /// Model reference
        model: String,

        /// Output directory for fly.toml
        #[arg(long, short = 'o', default_value = ".")]
        output: PathBuf,

        /// App name
        #[arg(long)]
        app: Option<String>,

        /// Region
        #[arg(long, default_value = "iad")]
        region: String,
    },

    /// Generate Cloudflare Workers deployment
    Cloudflare {
        /// Model reference
        model: String,

        /// Output directory for wrangler.toml
        #[arg(long, short = 'o', default_value = ".")]
        output: PathBuf,

        /// Worker name
        #[arg(long)]
        name: Option<String>,
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
            cmd_parf(
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
            format,
        } => {
            info!("Oracle Mode");
            cmd_oracle(
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
            )?;
        }
        Commands::Stack { command } => {
            info!("Stack Mode");
            cmd_stack(command)?;
        }
        Commands::Hf { command } => {
            info!("HuggingFace Mode");
            cmd_hf(command)?;
        }
        Commands::Pacha { command } => {
            info!("Pacha Model Registry Mode");
            pacha::cmd_pacha(command)?;
        }
        Commands::Data { command } => {
            info!("Data Platforms Mode");
            cmd_data(command)?;
        }
        Commands::Viz { command } => {
            info!("Visualization Frameworks Mode");
            cmd_viz(command)?;
        }
        Commands::Experiment { command } => {
            info!("Experiment Tracking Frameworks Mode");
            cmd_experiment(command)?;
        }
        Commands::Content { command } => {
            info!("Content Creation Tooling Mode");
            cmd_content(command)?;
        }
        Commands::Serve {
            model,
            host,
            port,
            openai_api,
            watch,
        } => {
            info!("Starting Model Server Mode");
            cmd_serve(model, &host, port, openai_api, watch)?;
        }
        Commands::Deploy { command } => {
            info!("Deployment Generation Mode");
            cmd_deploy(command)?;
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
    display_workflow_progress(&state);
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
        display_workflow_progress(state);
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

    display_workflow_progress(state);

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

    display_workflow_progress(state);

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
        display_workflow_progress(&state);
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

    // TODO: Implement actual optimization with Trueno
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
    display_workflow_progress(&state);

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
        display_workflow_progress(&state);
        return Ok(());
    }

    // Start validation phase
    state.start_phase(WorkflowPhase::Validation);
    state.save(&state_file)?;

    // Display validation settings
    println!("{}", "Validation Settings:".bright_yellow().bold());
    println!(
        "  {} Syscall tracing: {}",
        "•".bright_blue(),
        if trace_syscalls {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} Diff output: {}",
        "•".bright_blue(),
        if diff_output {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} Original tests: {}",
        "•".bright_blue(),
        if run_original_tests {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} Benchmarks: {}",
        "•".bright_blue(),
        if benchmark {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
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
            use crate::pipeline::{PipelineContext, PipelineStage, ValidationStage};

            let ctx = PipelineContext::new(PathBuf::from("."), PathBuf::from("."));

            let stage = ValidationStage::new(trace_syscalls, run_original_tests);

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
                        } else {
                            println!(
                                "{}",
                                "  ❌ Syscall traces differ - equivalence NOT verified".red()
                            );
                            validation_passed = false;
                        }
                    } else {
                        println!(
                            "{}",
                            "  ⚠️  Syscall tracing skipped (binaries not found)".yellow()
                        );
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
        state.fail_phase(
            WorkflowPhase::Validation,
            "Validation checks failed".to_string(),
        );
    }
    state.save(&state_file)?;

    // Display workflow progress
    display_workflow_progress(&state);

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
        display_workflow_progress(&state);
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

fn cmd_status() -> anyhow::Result<()> {
    println!("{}", "📊 Workflow Status".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|_| WorkflowState::new());

    // Check if any work has been done
    let has_started = state
        .phases
        .values()
        .any(|info| info.status != PhaseStatus::NotStarted);

    if !has_started {
        println!("{}", "No workflow started yet.".dimmed());
        println!();
        println!("{}", "💡 Get started:".bright_yellow().bold());
        println!(
            "  {} Run {} to analyze your project",
            "1.".bright_blue(),
            "batuta analyze".cyan()
        );
        println!(
            "  {} Run {} to initialize configuration",
            "2.".bright_blue(),
            "batuta init".cyan()
        );
        println!();
        return Ok(());
    }

    display_workflow_progress(&state);

    // Display detailed phase information
    println!("{}", "Phase Details:".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state
            .phases
            .get(&phase)
            .expect("workflow phase missing from state");

        let status_icon = match info.status {
            PhaseStatus::Completed => "✓".bright_green(),
            PhaseStatus::InProgress => "⏳".bright_yellow(),
            PhaseStatus::Failed => "✗".bright_red(),
            PhaseStatus::NotStarted => "○".dimmed(),
        };

        println!();
        println!("{} {}", status_icon, format!("{}", phase).bold());

        if let Some(started) = info.started_at {
            println!(
                "  Started: {}",
                started.format("%Y-%m-%d %H:%M:%S UTC").to_string().dimmed()
            );
        }

        if let Some(completed) = info.completed_at {
            println!(
                "  Completed: {}",
                completed
                    .format("%Y-%m-%d %H:%M:%S UTC")
                    .to_string()
                    .dimmed()
            );

            if let Some(started) = info.started_at {
                let duration = completed.signed_duration_since(started);
                println!(
                    "  Duration: {:.2}s",
                    duration.num_milliseconds() as f64 / 1000.0
                );
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
                println!(
                    "  Run {} to analyze your project",
                    "batuta analyze --languages --tdg".cyan()
                );
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
                println!(
                    "  Run {} to build final binary",
                    "batuta build --release".cyan()
                );
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
        println!(
            "  This will reset {} completed phase(s)",
            completed_count.to_string().yellow()
        );
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
    println!(
        "{}",
        "✅ Workflow state reset successfully!"
            .bright_green()
            .bold()
    );
    println!();
    println!("{}", "💡 Next Step:".bright_yellow().bold());
    println!(
        "  Run {} to start fresh",
        "batuta analyze --languages --tdg".cyan()
    );
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

#[allow(clippy::too_many_arguments)]
fn cmd_oracle(
    query: Option<String>,
    recommend: bool,
    problem: Option<String>,
    data_size: Option<String>,
    integrate: Option<String>,
    capabilities: Option<String>,
    list: bool,
    show: Option<String>,
    interactive: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::{OracleQuery, Recommender};

    let recommender = Recommender::new();

    // List all components
    if list {
        display_component_list(&recommender, format)?;
        return Ok(());
    }

    // Show component details
    if let Some(component_name) = show {
        display_component_details(&recommender, &component_name, format)?;
        return Ok(());
    }

    // Show capabilities
    if let Some(component_name) = capabilities {
        display_capabilities(&recommender, &component_name, format)?;
        return Ok(());
    }

    // Show integration pattern
    if let Some(components) = integrate {
        display_integration(&recommender, &components, format)?;
        return Ok(());
    }

    // Interactive mode
    if interactive {
        run_interactive_oracle(&recommender)?;
        return Ok(());
    }

    // Query mode
    if let Some(query_text) = query {
        // Parse data size if provided
        let parsed_size = data_size.and_then(|s| parse_data_size(&s));

        // Build query
        let mut oracle_query = OracleQuery::new(&query_text);
        if let Some(size) = parsed_size {
            oracle_query = oracle_query.with_data_size(size);
        }

        // Get recommendation
        let response = recommender.query_structured(&oracle_query);
        display_oracle_response(&response, format)?;
        return Ok(());
    }

    // Recommendation mode
    if recommend {
        let query_text = problem.unwrap_or_else(|| "general ML task".into());
        let response = recommender.query(&query_text);
        display_oracle_response(&response, format)?;
        return Ok(());
    }

    // Default: show help
    println!("{}", "🔮 Batuta Oracle Mode".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();
    println!(
        "{}",
        "Query the Sovereign AI Stack for recommendations".dimmed()
    );
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle".cyan(),
        "\"How do I train a random forest?\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recommend --problem".cyan(),
        "\"image classification\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --capabilities".cyan(),
        "aprender".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --integrate".cyan(),
        "\"aprender,realizar\"".dimmed()
    );
    println!("  {} {}", "batuta oracle --list".cyan(), "".dimmed());
    println!("  {} {}", "batuta oracle --interactive".cyan(), "".dimmed());
    println!();

    Ok(())
}

fn display_component_list(
    recommender: &oracle::Recommender,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "🔮 Sovereign AI Stack Components".bright_cyan().bold()
    );
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let components: Vec<_> = recommender.list_components();

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&components)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Sovereign AI Stack Components\n");
            println!("| Component | Version | Layer | Description |");
            println!("|-----------|---------|-------|-------------|");
            for name in &components {
                if let Some(comp) = recommender.get_component(name) {
                    println!(
                        "| {} | {} | {} | {} |",
                        comp.name, comp.version, comp.layer, comp.description
                    );
                }
            }
        }
        OracleOutputFormat::Text => {
            // Group by layer
            for layer in oracle::StackLayer::all() {
                let layer_components: Vec<_> = components
                    .iter()
                    .filter_map(|name| recommender.get_component(name))
                    .filter(|c| c.layer == layer)
                    .collect();

                if !layer_components.is_empty() {
                    println!("{} {}", "Layer".bold(), format!("{}", layer).cyan());
                    for comp in layer_components {
                        println!(
                            "  {} {} {} - {}",
                            "•".bright_blue(),
                            comp.name.bright_green(),
                            format!("v{}", comp.version).dimmed(),
                            comp.description.dimmed()
                        );
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn display_component_details(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let Some(comp) = recommender.get_component(name) else {
        println!("{} Component '{}' not found", "❌".red(), name);
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&comp)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## {}\n", comp.name);
            println!("**Version:** {}\n", comp.version);
            println!("**Layer:** {}\n", comp.layer);
            println!("**Description:** {}\n", comp.description);
            println!("### Capabilities\n");
            for cap in &comp.capabilities {
                println!(
                    "- **{}**{}",
                    cap.name,
                    cap.description
                        .as_ref()
                        .map(|d| format!(": {}", d))
                        .unwrap_or_default()
                );
            }
        }
        OracleOutputFormat::Text => {
            println!("{}", format!("📦 {}", comp.name).bright_cyan().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Version".bold(), comp.version.cyan());
            println!("{}: {}", "Layer".bold(), format!("{}", comp.layer).cyan());
            println!("{}: {}", "Description".bold(), comp.description);
            println!();
            println!("{}", "Capabilities:".bright_yellow());
            for cap in &comp.capabilities {
                let desc = cap
                    .description
                    .as_ref()
                    .map(|d| format!(" - {}", d.dimmed()))
                    .unwrap_or_default();
                println!("  {} {}{}", "•".bright_blue(), cap.name.green(), desc);
            }
            println!();
        }
    }

    Ok(())
}

fn display_capabilities(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let caps = recommender.get_capabilities(name);

    if caps.is_empty() {
        println!("{} No capabilities found for '{}'", "❌".red(), name);
        return Ok(());
    }

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&caps)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Capabilities of {}\n", name);
            for cap in &caps {
                println!("- {}", cap);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔧 Capabilities of {}", name).bright_cyan().bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            for cap in &caps {
                println!("  {} {}", "•".bright_blue(), cap.green());
            }
            println!();
        }
    }

    Ok(())
}

fn display_integration(
    recommender: &oracle::Recommender,
    components: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        println!(
            "{} Please specify two components separated by comma",
            "❌".red()
        );
        println!(
            "  Example: {} {}",
            "batuta oracle --integrate".cyan(),
            "\"aprender,realizar\"".dimmed()
        );
        return Ok(());
    }

    let from = parts[0];
    let to = parts[1];

    let Some(pattern) = recommender.get_integration(from, to) else {
        println!(
            "{} No integration pattern found from '{}' to '{}'",
            "❌".red(),
            from,
            to
        );
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&pattern)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Integration: {} → {}\n", from, to);
            println!("**Pattern:** {}\n", pattern.pattern_name);
            println!("**Description:** {}\n", pattern.description);
            if let Some(template) = &pattern.code_template {
                println!("### Code Example\n");
                println!("```rust\n{}\n```", template);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔗 Integration: {} → {}", from, to)
                    .bright_cyan()
                    .bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Pattern".bold(), pattern.pattern_name.cyan());
            println!("{}: {}", "Description".bold(), pattern.description);
            println!();
            if let Some(template) = &pattern.code_template {
                println!("{}", "Code Example:".bright_yellow());
                println!("{}", "─".repeat(40).dimmed());
                for line in template.lines() {
                    println!("  {}", line.dimmed());
                }
                println!("{}", "─".repeat(40).dimmed());
            }
            println!();
        }
    }

    Ok(())
}

fn display_oracle_response(
    response: &oracle::OracleResponse,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&response)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Oracle Recommendation\n");
            println!("**Problem Class:** {}\n", response.problem_class);
            if let Some(algo) = &response.algorithm {
                println!("**Algorithm:** {}\n", algo);
            }
            println!("### Primary Recommendation\n");
            println!("- **Component:** {}", response.primary.component);
            if let Some(path) = &response.primary.path {
                println!("- **Module:** `{}`", path);
            }
            println!(
                "- **Confidence:** {:.0}%",
                response.primary.confidence * 100.0
            );
            println!("- **Rationale:** {}\n", response.primary.rationale);

            if !response.supporting.is_empty() {
                println!("### Supporting Components\n");
                for rec in &response.supporting {
                    println!(
                        "- **{}** ({:.0}%): {}",
                        rec.component,
                        rec.confidence * 100.0,
                        rec.rationale
                    );
                }
                println!();
            }

            println!("### Compute Backend\n");
            println!("- **Backend:** {}", response.compute.backend);
            println!("- **Rationale:** {}\n", response.compute.rationale);

            if response.distribution.needed {
                println!("### Distribution\n");
                println!(
                    "- **Tool:** {}",
                    response.distribution.tool.as_deref().unwrap_or("N/A")
                );
                println!("- **Rationale:** {}\n", response.distribution.rationale);
            }

            if let Some(code) = &response.code_example {
                println!("### Code Example\n");
                println!("```rust\n{}\n```\n", code);
            }
        }
        OracleOutputFormat::Text => {
            println!();
            println!("{}", "🔮 Oracle Recommendation".bright_cyan().bold());
            println!("{}", "═".repeat(60).dimmed());
            println!();

            // Problem classification
            println!(
                "{} {}: {}",
                "📊".bright_blue(),
                "Problem Class".bold(),
                response.problem_class.cyan()
            );
            if let Some(algo) = &response.algorithm {
                println!(
                    "{} {}: {}",
                    "🧮".bright_blue(),
                    "Algorithm".bold(),
                    algo.cyan()
                );
            }
            println!();

            // Primary recommendation
            println!("{}", "🎯 Primary Recommendation".bright_yellow().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!(
                "  {}: {}",
                "Component".bold(),
                response.primary.component.bright_green()
            );
            if let Some(path) = &response.primary.path {
                println!("  {}: {}", "Module".bold(), path.cyan());
            }
            println!(
                "  {}: {}",
                "Confidence".bold(),
                format!("{:.0}%", response.primary.confidence * 100.0).bright_green()
            );
            println!(
                "  {}: {}",
                "Rationale".bold(),
                response.primary.rationale.dimmed()
            );
            println!();

            // Supporting components
            if !response.supporting.is_empty() {
                println!("{}", "🔧 Supporting Components".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                for rec in &response.supporting {
                    println!(
                        "  {} {} ({:.0}%)",
                        "•".bright_blue(),
                        rec.component.green(),
                        rec.confidence * 100.0
                    );
                    println!("    {}", rec.rationale.dimmed());
                }
                println!();
            }

            // Compute backend
            println!("{}", "⚡ Compute Backend".bright_yellow().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!(
                "  {}: {}",
                "Backend".bold(),
                format!("{}", response.compute.backend).bright_green()
            );
            println!("  {}", response.compute.rationale.dimmed());
            println!();

            // Distribution
            if response.distribution.needed {
                println!("{}", "🌐 Distribution".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                println!(
                    "  {}: {}",
                    "Tool".bold(),
                    response
                        .distribution
                        .tool
                        .as_deref()
                        .unwrap_or("N/A")
                        .bright_green()
                );
                if let Some(nodes) = response.distribution.node_count {
                    println!("  {}: {}", "Nodes".bold(), nodes);
                }
                println!("  {}", response.distribution.rationale.dimmed());
                println!();
            }

            // Code example
            if let Some(code) = &response.code_example {
                println!("{}", "💡 Example Code".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                for line in code.lines() {
                    println!("  {}", line.dimmed());
                }
                println!("{}", "─".repeat(50).dimmed());
                println!();
            }

            // Related queries
            if !response.related_queries.is_empty() {
                println!("{}", "❓ Related Queries".bright_yellow());
                for query in &response.related_queries {
                    println!("  {} {}", "→".bright_blue(), query.dimmed());
                }
                println!();
            }

            println!("{}", "═".repeat(60).dimmed());
        }
    }

    Ok(())
}

fn run_interactive_oracle(recommender: &oracle::Recommender) -> anyhow::Result<()> {
    use std::io::{self, Write};

    println!();
    println!("{}", "🔮 Batuta Oracle Mode v1.0".bright_cyan().bold());
    println!(
        "{}",
        "   Ask questions about the Sovereign AI Stack".dimmed()
    );
    println!("{}", "   Type 'exit' or 'quit' to leave".dimmed());
    println!();

    loop {
        print!("{} ", ">".bright_cyan());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            println!();
            println!("{}", "👋 Goodbye!".bright_cyan());
            break;
        }

        if input == "help" {
            println!();
            println!("{}", "Commands:".bright_yellow());
            println!("  {} - Ask a question about the stack", "any text".cyan());
            println!("  {} - List all components", "list".cyan());
            println!("  {} - Show component details", "show <component>".cyan());
            println!("  {} - Show capabilities", "caps <component>".cyan());
            println!("  {} - Exit interactive mode", "exit".cyan());
            println!();
            continue;
        }

        if input == "list" {
            display_component_list(recommender, OracleOutputFormat::Text)?;
            continue;
        }

        if input.starts_with("show ") {
            let name = input
                .strip_prefix("show ")
                .expect("prefix verified by starts_with")
                .trim();
            display_component_details(recommender, name, OracleOutputFormat::Text)?;
            continue;
        }

        if input.starts_with("caps ") {
            let name = input
                .strip_prefix("caps ")
                .expect("prefix verified by starts_with")
                .trim();
            display_capabilities(recommender, name, OracleOutputFormat::Text)?;
            continue;
        }

        // Process as query
        let response = recommender.query(input);
        display_oracle_response(&response, OracleOutputFormat::Text)?;
    }

    Ok(())
}

fn parse_data_size(s: &str) -> Option<oracle::DataSize> {
    cli::parse_data_size_value(s).map(oracle::DataSize::samples)
}

// ============================================================================
// Stack Command Implementation
// ============================================================================

fn cmd_stack(command: StackCommand) -> anyhow::Result<()> {
    match command {
        StackCommand::Check {
            project,
            format,
            strict,
            verify_published,
            offline,
            workspace,
        } => {
            cmd_stack_check(
                project,
                format,
                strict,
                verify_published,
                offline,
                workspace,
            )?;
        }
        StackCommand::Release {
            crate_name,
            all,
            dry_run,
            bump,
            no_verify,
            yes,
            publish,
        } => {
            cmd_stack_release(crate_name, all, dry_run, bump, no_verify, yes, publish)?;
        }
        StackCommand::Status {
            simple,
            format,
            tree,
        } => {
            cmd_stack_status(simple, format, tree)?;
        }
        StackCommand::Sync {
            crate_name,
            all,
            dry_run,
            align,
        } => {
            cmd_stack_sync(crate_name, all, dry_run, align)?;
        }
        StackCommand::Tree {
            format,
            health,
            filter,
        } => {
            cmd_stack_tree(&format, health, filter.as_deref())?;
        }
        StackCommand::Quality {
            component,
            strict,
            format,
            verify_hero: _,
            verbose: _,
            workspace,
        } => {
            cmd_stack_quality(component, strict, format, workspace)?;
        }
        StackCommand::Gate { workspace, quiet } => {
            cmd_stack_gate(workspace, quiet)?;
        }
    }
    Ok(())
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
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker, StackQualityReport};

    println!("{}", "📦 PAIML Stack Release".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // QUALITY GATE: Enforce A- threshold before release (unless --no-verify)
    if !no_verify {
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
                    if let Ok(quality) =
                        rt.block_on(async { checker.check_component(comp_name).await })
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
            println!("Fix quality issues before releasing, or use --no-verify to skip (not recommended).");
            anyhow::bail!(
                "Release blocked: {} component(s) below A- threshold",
                report.blocked_components.len()
            );
        }

        println!("{}", "✅ Quality gate passed".bright_green());
        println!();
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

    // Plan release
    let plan = if all {
        orchestrator.plan_all_releases()?
    } else if let Some(name) = crate_name {
        orchestrator.plan_release(&name)?
    } else {
        println!("{}", "❌ Specify a crate name or use --all".red());
        return Ok(());
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
        format_quality_report_json, format_quality_report_text, tree::LAYER_DEFINITIONS,
        QualityChecker, StackQualityReport,
    };

    // Workspace is the parent directory containing all stack crates
    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap())
    });

    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    let report = if let Some(comp_name) = component {
        // Check single component
        let checker = QualityChecker::new(workspace_path.clone());
        let quality = rt.block_on(async { checker.check_component(&comp_name).await })?;
        StackQualityReport::from_components(vec![quality])
    } else {
        // Check all components from LAYER_DEFINITIONS
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

        StackQualityReport::from_components(components)
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

/// Quality gate enforcement for CI/pre-commit hooks
///
/// Checks all downstream PAIML stack components and fails if any are below A- threshold.
/// This is designed to be used in pre-commit hooks or CI pipelines to prevent
/// commits/deployments when quality standards are not met.
fn cmd_stack_gate(workspace: Option<PathBuf>, quiet: bool) -> anyhow::Result<()> {
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker, StackQualityReport};

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

    // Check all components from LAYER_DEFINITIONS
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

// ============================================================================
// HuggingFace Command Implementation
// ============================================================================

fn cmd_hf(command: HfCommand) -> anyhow::Result<()> {
    match command {
        HfCommand::Tree {
            integration,
            format,
        } => {
            cmd_hf_tree(integration, &format)?;
        }
        HfCommand::Search {
            asset_type,
            query,
            task,
            limit,
        } => {
            cmd_hf_search(asset_type, &query, task.as_deref(), limit)?;
        }
        HfCommand::Info {
            asset_type,
            repo_id,
        } => {
            cmd_hf_info(asset_type, &repo_id)?;
        }
        HfCommand::Pull {
            asset_type,
            repo_id,
            output,
            quantization,
        } => {
            cmd_hf_pull(asset_type, &repo_id, output, quantization)?;
        }
        HfCommand::Push {
            asset_type,
            path,
            repo,
            message,
        } => {
            cmd_hf_push(asset_type, &path, &repo, &message)?;
        }
    }
    Ok(())
}

fn cmd_hf_tree(integration: bool, format: &str) -> anyhow::Result<()> {
    use hf::tree::{
        build_hf_tree, build_integration_tree, format_hf_tree_ascii, format_hf_tree_json,
        format_integration_tree_ascii, format_integration_tree_json,
    };

    let output = if integration {
        let tree = build_integration_tree();
        match format {
            "json" => format_integration_tree_json(&tree)?,
            _ => format_integration_tree_ascii(&tree),
        }
    } else {
        let tree = build_hf_tree();
        match format {
            "json" => format_hf_tree_json(&tree)?,
            _ => format_hf_tree_ascii(&tree),
        }
    };

    println!("{}", output);
    Ok(())
}

fn cmd_hf_search(
    asset_type: HfAssetType,
    query: &str,
    task: Option<&str>,
    limit: usize,
) -> anyhow::Result<()> {
    println!("{}", "🔍 HuggingFace Hub Search".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "models",
        HfAssetType::Dataset => "datasets",
        HfAssetType::Space => "spaces",
    };

    println!("Searching {} for: {}", type_str.cyan(), query.yellow());
    if let Some(t) = task {
        println!("Task filter: {}", t.cyan());
    }
    println!("Limit: {}", limit);
    println!();

    // Hub API search integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will query: https://huggingface.co/api/{}", type_str);

    Ok(())
}

fn cmd_hf_info(asset_type: HfAssetType, repo_id: &str) -> anyhow::Result<()> {
    println!("{}", "📋 HuggingFace Asset Info".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Repository: {}", repo_id.yellow());
    println!();

    // Hub API info integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!(
        "Will fetch: https://huggingface.co/api/{}s/{}",
        type_str, repo_id
    );

    Ok(())
}

fn cmd_hf_pull(
    asset_type: HfAssetType,
    repo_id: &str,
    output: Option<PathBuf>,
    quantization: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "⬇️  HuggingFace Pull".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Repository: {}", repo_id.yellow());
    if let Some(ref out) = output {
        println!("Output: {}", out.display());
    }
    if let Some(ref q) = quantization {
        println!("Quantization: {}", q.cyan());
    }
    println!();

    // Hub API download integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will download from: https://huggingface.co/{}", repo_id);

    Ok(())
}

fn cmd_hf_push(
    asset_type: HfAssetType,
    path: &Path,
    repo: &str,
    message: &str,
) -> anyhow::Result<()> {
    println!("{}", "⬆️  HuggingFace Push".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let type_str = match asset_type {
        HfAssetType::Model => "model",
        HfAssetType::Dataset => "dataset",
        HfAssetType::Space => "space",
    };

    println!("Type: {}", type_str.cyan());
    println!("Local path: {}", path.display());
    println!("Target repo: {}", repo.yellow());
    println!("Message: {}", message);
    println!();

    // Hub API upload integration planned for v0.2
    // See roadmap item HUB-API for implementation plan
    println!(
        "{}",
        "⚠️  Hub API integration not yet implemented.".yellow()
    );
    println!("Will push to: https://huggingface.co/{}", repo);

    Ok(())
}

// ============================================================================
// Data Platforms Command Implementation
// ============================================================================

fn cmd_data(command: DataCommand) -> anyhow::Result<()> {
    match command {
        DataCommand::Tree {
            platform,
            integration,
            format,
        } => {
            cmd_data_tree(platform.as_deref(), integration, &format)?;
        }
    }
    Ok(())
}

fn cmd_data_tree(platform: Option<&str>, integration: bool, format: &str) -> anyhow::Result<()> {
    use data::tree::{
        build_aws_tree, build_databricks_tree, build_huggingface_tree, build_integration_mappings,
        build_snowflake_tree, format_all_platforms, format_integration_mappings,
        format_platform_tree,
    };

    if integration {
        // Show PAIML integration mappings
        let output = match format {
            "json" => {
                let mappings = build_integration_mappings();
                serde_json::to_string_pretty(&mappings)?
            }
            _ => format_integration_mappings(),
        };
        println!("{}", output);
    } else if let Some(platform_name) = platform {
        // Show specific platform tree
        let platform = platform_name.to_lowercase();
        let tree = match platform.as_str() {
            "databricks" => build_databricks_tree(),
            "snowflake" => build_snowflake_tree(),
            "aws" => build_aws_tree(),
            "huggingface" | "hf" => build_huggingface_tree(),
            _ => {
                anyhow::bail!(
                    "Unknown platform: {}. Valid options: databricks, snowflake, aws, huggingface",
                    platform_name
                );
            }
        };
        let output = match format {
            "json" => serde_json::to_string_pretty(&tree)?,
            _ => format_platform_tree(&tree),
        };
        println!("{}", output);
    } else {
        // Show all platforms
        let output = match format {
            "json" => {
                let trees = vec![
                    build_databricks_tree(),
                    build_snowflake_tree(),
                    build_aws_tree(),
                    build_huggingface_tree(),
                ];
                serde_json::to_string_pretty(&trees)?
            }
            _ => format_all_platforms(),
        };
        println!("{}", output);
    }

    Ok(())
}

// ============================================================================
// Visualization Frameworks Command Implementation
// ============================================================================

fn cmd_viz(command: VizCommand) -> anyhow::Result<()> {
    match command {
        VizCommand::Tree {
            framework,
            integration,
            format,
        } => {
            cmd_viz_tree(framework.as_deref(), integration, &format)?;
        }
        VizCommand::Dashboard {
            source,
            port,
            theme,
            output,
        } => {
            cmd_viz_dashboard(&source, port, &theme, output.as_deref())?;
        }
    }
    Ok(())
}

fn cmd_viz_tree(framework: Option<&str>, integration: bool, format: &str) -> anyhow::Result<()> {
    use viz::tree::{
        build_dash_tree, build_gradio_tree, build_integration_mappings, build_panel_tree,
        build_streamlit_tree, format_all_frameworks, format_framework_tree,
        format_integration_mappings,
    };

    if integration {
        // Show PAIML replacement mappings
        let output = match format {
            "json" => {
                let mappings = build_integration_mappings();
                serde_json::to_string_pretty(&mappings)?
            }
            _ => format_integration_mappings(),
        };
        println!("{}", output);
    } else if let Some(framework_name) = framework {
        // Show specific framework tree
        let fw = framework_name.to_lowercase();
        let tree = match fw.as_str() {
            "gradio" => build_gradio_tree(),
            "streamlit" => build_streamlit_tree(),
            "panel" => build_panel_tree(),
            "dash" => build_dash_tree(),
            _ => {
                anyhow::bail!(
                    "Unknown framework: {}. Valid options: gradio, streamlit, panel, dash",
                    framework_name
                );
            }
        };
        let output = match format {
            "json" => serde_json::to_string_pretty(&tree)?,
            _ => format_framework_tree(&tree),
        };
        println!("{}", output);
    } else {
        // Show all frameworks
        let output = match format {
            "json" => {
                let trees = vec![
                    build_gradio_tree(),
                    build_streamlit_tree(),
                    build_panel_tree(),
                    build_dash_tree(),
                ];
                serde_json::to_string_pretty(&trees)?
            }
            _ => format_all_frameworks(),
        };
        println!("{}", output);
    }

    Ok(())
}

/// Generate Presentar dashboard configuration for monitoring
fn cmd_viz_dashboard(
    source: &str,
    port: u16,
    theme: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    // Parse data source URI
    let (source_type, source_path) = if let Some(rest) = source.strip_prefix("trueno-db://") {
        ("trueno-db", rest)
    } else if let Some(rest) = source.strip_prefix("prometheus://") {
        ("prometheus", rest)
    } else {
        ("file", source)
    };

    // Generate Presentar dashboard configuration
    let config = format!(
        r#"# Presentar Monitoring Dashboard
# Generated by: batuta viz dashboard

app:
  name: "Realizar Monitoring"
  version: "1.0.0"
  port: {port}
  theme: "{theme}"

data_source:
  type: "{source_type}"
  path: "{source_path}"
  refresh_interval_ms: 1000

panels:
  - id: "inference_latency"
    title: "Inference Latency"
    type: "timeseries"
    query: |
      SELECT time, p50, p95, p99
      FROM realizar_metrics
      WHERE metric = 'inference_latency_ms'
      ORDER BY time DESC
      LIMIT 100
    y_axis: "Latency (ms)"

  - id: "throughput"
    title: "Token Throughput"
    type: "gauge"
    query: |
      SELECT avg(tokens_per_second) as tps
      FROM realizar_metrics
      WHERE metric = 'throughput'
      AND time > now() - interval '1 minute'
    max: 1000
    thresholds:
      - value: 100
        color: "red"
      - value: 500
        color: "yellow"
      - value: 800
        color: "green"

  - id: "model_requests"
    title: "Requests by Model"
    type: "bar"
    query: |
      SELECT model_name, count(*) as requests
      FROM realizar_metrics
      WHERE metric = 'request_count'
      GROUP BY model_name
      ORDER BY requests DESC
      LIMIT 10

  - id: "error_rate"
    title: "Error Rate"
    type: "stat"
    query: |
      SELECT
        (count(*) FILTER (WHERE status = 'error')) * 100.0 / count(*) as error_pct
      FROM realizar_metrics
      WHERE time > now() - interval '5 minutes'
    unit: "%"
    thresholds:
      - value: 1
        color: "green"
      - value: 5
        color: "yellow"
      - value: 10
        color: "red"

  - id: "ab_tests"
    title: "A/B Test Results"
    type: "table"
    query: |
      SELECT
        test_name,
        variant,
        requests,
        success_rate,
        avg_latency_ms
      FROM ab_test_results
      ORDER BY test_name, variant

layout:
  rows:
    - height: "300px"
      panels: ["inference_latency", "throughput"]
    - height: "250px"
      panels: ["model_requests", "error_rate"]
    - height: "200px"
      panels: ["ab_tests"]
"#,
        port = port,
        theme = theme,
        source_type = source_type,
        source_path = source_path,
    );

    if let Some(output_path) = output {
        std::fs::write(output_path, &config)?;
        println!("Dashboard config written to: {}", output_path);
    } else {
        println!("{}", config);
        println!();
        println!("{}", "To launch dashboard:".cyan());
        println!("  presentar serve dashboard.yaml --port {}", port);
    }

    Ok(())
}

fn cmd_experiment(command: ExperimentCommand) -> anyhow::Result<()> {
    match command {
        ExperimentCommand::Tree {
            framework,
            integration,
            format,
        } => {
            cmd_experiment_tree(framework.as_deref(), integration, &format)?;
        }
    }
    Ok(())
}

fn cmd_experiment_tree(
    framework: Option<&str>,
    integration: bool,
    format: &str,
) -> anyhow::Result<()> {
    use experiment::tree::{
        build_dvc_tree, build_integration_mappings, build_mlflow_tree, build_neptune_tree,
        build_wandb_tree, format_all_frameworks, format_framework_tree,
        format_integration_mappings,
    };

    if integration {
        // Show PAIML replacement mappings
        let output = match format {
            "json" => {
                let mappings = build_integration_mappings();
                serde_json::to_string_pretty(&mappings)?
            }
            _ => format_integration_mappings(),
        };
        println!("{}", output);
    } else if let Some(framework_name) = framework {
        // Show specific framework tree
        let fw = framework_name.to_lowercase();
        let tree = match fw.as_str() {
            "mlflow" => build_mlflow_tree(),
            "wandb" => build_wandb_tree(),
            "neptune" => build_neptune_tree(),
            "dvc" => build_dvc_tree(),
            _ => {
                anyhow::bail!(
                    "Unknown framework: {}. Valid options: mlflow, wandb, neptune, dvc",
                    framework_name
                );
            }
        };
        let output = match format {
            "json" => serde_json::to_string_pretty(&tree)?,
            _ => format_framework_tree(&tree),
        };
        println!("{}", output);
    } else {
        // Show all frameworks
        let output = match format {
            "json" => {
                let trees = vec![
                    build_mlflow_tree(),
                    build_wandb_tree(),
                    build_neptune_tree(),
                    build_dvc_tree(),
                ];
                serde_json::to_string_pretty(&trees)?
            }
            _ => format_all_frameworks(),
        };
        println!("{}", output);
    }

    Ok(())
}

fn cmd_content(command: ContentCommand) -> anyhow::Result<()> {
    match command {
        ContentCommand::Emit {
            r#type,
            title,
            audience,
            word_count,
            source_context,
            show_budget,
            output,
            level,
        } => {
            cmd_content_emit(
                &r#type,
                title,
                audience,
                word_count,
                source_context,
                show_budget,
                output,
                level,
            )?;
        }
        ContentCommand::Validate {
            r#type,
            file,
            llm_judge,
        } => {
            cmd_content_validate(&r#type, &file, llm_judge)?;
        }
        ContentCommand::Types => {
            cmd_content_types()?;
        }
    }
    Ok(())
}

fn cmd_content_emit(
    content_type: &str,
    title: Option<String>,
    audience: Option<String>,
    word_count: Option<usize>,
    _source_context: Option<String>,
    show_budget: bool,
    output: Option<std::path::PathBuf>,
    level: Option<String>,
) -> anyhow::Result<()> {
    use content::{ContentType, CourseLevel, EmitConfig, PromptEmitter};

    let ct = ContentType::from_str(content_type).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut config = EmitConfig::new(ct);
    if let Some(t) = title {
        config = config.with_title(t);
    }
    if let Some(a) = audience {
        config = config.with_audience(a);
    }
    if let Some(wc) = word_count {
        config = config.with_word_count(wc);
    }
    if let Some(lvl) = level {
        let course_level: CourseLevel = lvl
            .parse()
            .map_err(|e: content::ContentError| anyhow::anyhow!("{}", e))?;
        config = config.with_course_level(course_level);
    }
    config.show_budget = show_budget;

    let emitter = PromptEmitter::new();
    let prompt = emitter
        .emit(&config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if let Some(path) = output {
        std::fs::write(&path, &prompt)?;
        println!(
            "{}",
            format!("Prompt written to: {}", path.display()).green()
        );
    } else {
        println!("{}", prompt);
    }

    Ok(())
}

fn cmd_content_validate(
    content_type: &str,
    file: &std::path::Path,
    _llm_judge: bool,
) -> anyhow::Result<()> {
    use content::{ContentType, ContentValidator};

    let ct = ContentType::from_str(content_type).map_err(|e| anyhow::anyhow!("{}", e))?;

    let content = std::fs::read_to_string(file)?;
    let validator = ContentValidator::new(ct);
    let result = validator.validate(&content);

    println!(
        "{}",
        format!("Validating {} as {}...", file.display(), ct.name()).cyan()
    );
    println!();
    println!("{}", result.format_display());

    if result.has_critical() || result.has_errors() {
        anyhow::bail!("Validation failed with critical errors");
    }

    Ok(())
}

fn cmd_content_types() -> anyhow::Result<()> {
    use content::ContentType;

    println!("{}", "Content Types".bright_cyan().bold());
    println!("{}", "=".repeat(60));
    println!();
    println!(
        "{:<6} {:<22} {:<20} Target Length",
        "Code", "Name", "Output Format"
    );
    println!("{}", "-".repeat(60));

    for ct in ContentType::all() {
        let range = ct.target_length();
        let length_str = if range.start == 0 && range.end == 0 {
            "N/A".to_string()
        } else {
            let unit = if matches!(
                ct,
                ContentType::HighLevelOutline | ContentType::DetailedOutline
            ) {
                "lines"
            } else {
                "words"
            };
            format!("{}-{} {}", range.start, range.end, unit)
        };
        println!(
            "{:<6} {:<22} {:<20} {}",
            ct.code(),
            ct.name(),
            ct.output_format(),
            length_str
        );
    }

    println!();
    println!("{}", "Usage:".bright_yellow());
    println!("  batuta content emit -t hlo --title \"My Course\"");
    println!("  batuta content emit -t bch --title \"Chapter 1\" --word-count 4000");
    println!("  batuta content validate -t bch chapter.md");

    Ok(())
}

// ============================================================================
// Serve Command Implementation (per spec §5)
// ============================================================================

fn cmd_serve(
    model: Option<String>,
    host: &str,
    port: u16,
    openai_api: bool,
    watch: bool,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "🚀 Starting Realizar Model Server".bright_cyan().bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Resolve model reference if provided
    let resolved_model = if let Some(model_ref) = &model {
        // Try to resolve via pacha aliases
        let resolved = resolve_model_for_serve(model_ref);
        println!("{} Model: {}", "•".bright_blue(), model_ref.cyan());
        if resolved != *model_ref {
            println!("{} Resolved: {}", "•".bright_blue(), resolved.dimmed());
        }
        Some(resolved)
    } else {
        println!(
            "{} Model: {}",
            "•".bright_blue(),
            "demo (no model specified)".dimmed()
        );
        None
    };

    println!(
        "{} Address: {}:{}",
        "•".bright_blue(),
        host.cyan(),
        port.to_string().cyan()
    );
    println!(
        "{} OpenAI API: {}",
        "•".bright_blue(),
        if openai_api {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    if watch {
        println!("{} Hot-reload: {}", "•".bright_blue(), "enabled".green());
    }
    println!();

    // Check if model is cached (if using pacha scheme)
    if let Some(ref resolved) = resolved_model {
        if resolved.starts_with("hf://") || resolved.starts_with("pacha://") {
            println!("{}", "Checking model cache...".dimmed());
            println!(
                "{} Model will be pulled on first request if not cached",
                "ℹ".bright_blue()
            );
            println!();
        }
    }

    println!("{}", "Endpoints:".bright_yellow());
    println!("  GET  /health              - Health check");
    println!("  GET  /metrics             - Prometheus metrics");
    println!("  POST /generate            - Text generation");
    println!("  POST /tokenize            - Tokenize text");
    println!("  POST /stream/generate     - Streaming generation (SSE)");
    if openai_api {
        println!();
        println!("{}", "OpenAI-Compatible API:".bright_yellow());
        println!("  GET  /v1/models           - List models");
        println!("  POST /v1/chat/completions - Chat completions");
    }
    println!();

    // Show curl examples
    println!("{}", "Quick Test:".bright_yellow());
    println!("  # Health check");
    println!("  curl http://{}:{}/health", host, port);
    println!();
    if openai_api {
        println!("  # Chat completion (OpenAI-compatible)");
        println!("  curl http://{}:{}/v1/chat/completions \\", host, port);
        println!("    -H \"Content-Type: application/json\" \\");
        println!("    -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'");
        println!();
    }

    // Note: Full integration with Realizar would require tokio runtime
    println!("{}", "─".repeat(60).dimmed());
    println!("{}", "Note:".bright_yellow());
    println!("  For production serving, use the Realizar CLI directly:");
    println!();
    if let Some(ref model_ref) = resolved_model {
        println!(
            "  {} {}",
            "realizar serve --model".cyan(),
            model_ref.bright_white()
        );
    } else {
        println!("  {} ", "realizar serve --demo".cyan());
    }
    println!();

    // Show pacha model management
    println!("{}", "Model Management:".bright_yellow());
    println!("  # Pull a model first");
    println!(
        "  batuta pacha pull {}",
        model.as_deref().unwrap_or("llama3:8b")
    );
    println!();
    println!("  # List cached models");
    println!("  batuta pacha list");

    Ok(())
}

/// Resolve model reference for serving (alias expansion)
fn resolve_model_for_serve(model_ref: &str) -> String {
    // Built-in aliases for common models
    let aliases = [
        ("llama3", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        ("llama3:8b", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        (
            "llama3:70b",
            "hf://meta-llama/Meta-Llama-3-70B-Instruct-GGUF",
        ),
        ("mistral", "hf://mistralai/Mistral-7B-Instruct-v0.2-GGUF"),
        ("mixtral", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        ("phi3", "hf://microsoft/Phi-3-mini-4k-instruct-gguf"),
        ("gemma", "hf://google/gemma-7b-it-GGUF"),
        ("qwen2", "hf://Qwen/Qwen2-7B-Instruct-GGUF"),
        ("codellama", "hf://codellama/CodeLlama-7b-Instruct-GGUF"),
    ];

    for (alias, target) in &aliases {
        if model_ref == *alias {
            return target.to_string();
        }
    }

    // If already a full URI, return as-is
    if model_ref.contains("://") {
        return model_ref.to_string();
    }

    // Otherwise, assume pacha:// scheme
    format!("pacha://{}", model_ref)
}

// ============================================================================
// Deploy Command Implementation (per spec §6)
// ============================================================================

fn cmd_deploy(command: DeployCommand) -> anyhow::Result<()> {
    match command {
        DeployCommand::Docker {
            model,
            output,
            base_image,
            port,
            multi_stage,
        } => {
            cmd_deploy_docker(&model, &output, &base_image, port, multi_stage)?;
        }
        DeployCommand::Lambda {
            model,
            output,
            memory,
            timeout,
            sam,
        } => {
            cmd_deploy_lambda(&model, &output, memory, timeout, sam)?;
        }
        DeployCommand::K8s {
            model,
            output,
            replicas,
            namespace,
            helm,
            hpa,
        } => {
            cmd_deploy_k8s(&model, &output, replicas, &namespace, helm, hpa)?;
        }
        DeployCommand::Fly {
            model,
            output,
            app,
            region,
        } => {
            cmd_deploy_fly(&model, &output, app.as_deref(), &region)?;
        }
        DeployCommand::Cloudflare {
            model,
            output,
            name,
        } => {
            cmd_deploy_cloudflare(&model, &output, name.as_deref())?;
        }
    }
    Ok(())
}

fn cmd_deploy_docker(
    model: &str,
    output: &Path,
    base_image: &str,
    port: u16,
    multi_stage: bool,
) -> anyhow::Result<()> {
    println!("{}", "🐳 Generating Docker Deployment".bright_cyan().bold());
    println!();
    println!("{} Model: {}", "•".bright_blue(), model.cyan());
    println!("{} Output: {}", "•".bright_blue(), output.display());
    println!("{} Base image: {}", "•".bright_blue(), base_image.cyan());
    println!("{} Port: {}", "•".bright_blue(), port);
    println!(
        "{} Multi-stage: {}",
        "•".bright_blue(),
        if multi_stage {
            "yes".green()
        } else {
            "no".dimmed()
        }
    );
    println!();

    // Generate Dockerfile
    let dockerfile = if multi_stage {
        format!(
            r#"# Multi-stage Dockerfile for Realizar model serving
# Generated by batuta deploy docker

# Build stage
FROM rust:latest AS builder
WORKDIR /app

# Install dependencies
RUN cargo install realizar

# Runtime stage
FROM {base_image}
WORKDIR /app

# Copy binary from builder
COPY --from=builder /usr/local/cargo/bin/realizar /usr/local/bin/

# Copy model (if local)
# COPY model.gguf /app/model.gguf

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:{port}/health || exit 1

# Run server
ENV MODEL_REF="{model}"
CMD ["realizar", "serve", "--host", "0.0.0.0", "--port", "{port}"]
"#
        )
    } else {
        format!(
            r#"# Dockerfile for Realizar model serving
# Generated by batuta deploy docker

FROM {base_image}
WORKDIR /app

# Install realizar
RUN cargo install realizar

# Copy model (if local)
# COPY model.gguf /app/model.gguf

EXPOSE {port}

ENV MODEL_REF="{model}"
CMD ["realizar", "serve", "--host", "0.0.0.0", "--port", "{port}"]
"#
        )
    };

    let dockerfile_path = output.join("Dockerfile");
    std::fs::write(&dockerfile_path, dockerfile)?;

    println!(
        "{} Generated: {}",
        "✓".bright_green(),
        dockerfile_path.display()
    );
    println!();
    println!("{}", "Build and run:".bright_yellow());
    println!("  docker build -t my-model-server .");
    println!("  docker run -p {}:{} my-model-server", port, port);

    Ok(())
}

fn cmd_deploy_lambda(
    model: &str,
    output: &Path,
    memory: u32,
    timeout: u32,
    sam: bool,
) -> anyhow::Result<()> {
    println!("{}", "λ Generating Lambda Deployment".bright_cyan().bold());
    println!();
    println!("{} Model: {}", "•".bright_blue(), model.cyan());
    println!("{} Output: {}", "•".bright_blue(), output.display());
    println!("{} Memory: {} MB", "•".bright_blue(), memory);
    println!("{} Timeout: {} seconds", "•".bright_blue(), timeout);
    println!(
        "{} SAM template: {}",
        "•".bright_blue(),
        if sam { "yes".green() } else { "no".dimmed() }
    );
    println!();

    if sam {
        let template = format!(
            r#"AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Realizar model serving Lambda
# Generated by batuta deploy lambda

Globals:
  Function:
    Timeout: {timeout}
    MemorySize: {memory}
    Runtime: provided.al2
    Architectures:
      - arm64

Resources:
  ModelFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: bootstrap
      CodeUri: .
      Description: Realizar model inference
      Events:
        Inference:
          Type: Api
          Properties:
            Path: /v1/chat/completions
            Method: post
        Health:
          Type: Api
          Properties:
            Path: /health
            Method: get
      Environment:
        Variables:
          MODEL_REF: "{model}"

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${{ServerlessRestApi}}.execute-api.${{AWS::Region}}.amazonaws.com/Prod/"
"#
        );

        let template_path = output.join("template.yaml");
        std::fs::write(&template_path, template)?;
        println!(
            "{} Generated: {}",
            "✓".bright_green(),
            template_path.display()
        );
    }

    println!();
    println!("{}", "Build for Lambda:".bright_yellow());
    println!("  cargo lambda build --release --arm64");
    if sam {
        println!();
        println!("{}", "Deploy with SAM:".bright_yellow());
        println!("  sam deploy --guided");
    }

    Ok(())
}

fn cmd_deploy_k8s(
    model: &str,
    output: &Path,
    replicas: u32,
    namespace: &str,
    helm: bool,
    hpa: bool,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "☸ Generating Kubernetes Deployment".bright_cyan().bold()
    );
    println!();
    println!("{} Model: {}", "•".bright_blue(), model.cyan());
    println!("{} Output: {}", "•".bright_blue(), output.display());
    println!("{} Replicas: {}", "•".bright_blue(), replicas);
    println!("{} Namespace: {}", "•".bright_blue(), namespace.cyan());
    println!(
        "{} Helm chart: {}",
        "•".bright_blue(),
        if helm { "yes".green() } else { "no".dimmed() }
    );
    println!(
        "{} HPA: {}",
        "•".bright_blue(),
        if hpa {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!();

    let deployment = format!(
        r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: realizar-model-server
  namespace: {namespace}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: realizar-model-server
  template:
    metadata:
      labels:
        app: realizar-model-server
    spec:
      containers:
      - name: realizar
        image: realizar:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_REF
          value: "{model}"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: realizar-model-server
  namespace: {namespace}
spec:
  selector:
    app: realizar-model-server
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
"#
    );

    let deployment_path = output.join("deployment.yaml");
    std::fs::write(&deployment_path, deployment)?;
    println!(
        "{} Generated: {}",
        "✓".bright_green(),
        deployment_path.display()
    );

    if hpa {
        let hpa_manifest = format!(
            r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: realizar-model-server-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: realizar-model-server
  minReplicas: {replicas}
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"#
        );
        let hpa_path = output.join("hpa.yaml");
        std::fs::write(&hpa_path, hpa_manifest)?;
        println!("{} Generated: {}", "✓".bright_green(), hpa_path.display());
    }

    println!();
    println!("{}", "Deploy:".bright_yellow());
    println!("  kubectl apply -f {}", deployment_path.display());
    if hpa {
        println!("  kubectl apply -f {}", output.join("hpa.yaml").display());
    }

    Ok(())
}

fn cmd_deploy_fly(
    model: &str,
    output: &Path,
    app: Option<&str>,
    region: &str,
) -> anyhow::Result<()> {
    println!("{}", "✈️ Generating Fly.io Deployment".bright_cyan().bold());
    println!();
    println!("{} Model: {}", "•".bright_blue(), model.cyan());
    println!("{} Output: {}", "•".bright_blue(), output.display());
    println!(
        "{} App: {}",
        "•".bright_blue(),
        app.unwrap_or("auto-generated").cyan()
    );
    println!("{} Region: {}", "•".bright_blue(), region.cyan());
    println!();

    let app_name = app.unwrap_or("realizar-model-server");
    let fly_toml = format!(
        r#"# fly.toml - Fly.io configuration
# Generated by batuta deploy fly

app = "{app_name}"
primary_region = "{region}"

[env]
  MODEL_REF = "{model}"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [[services.tcp_checks]]
    grace_period = "30s"
    interval = "15s"
    restart_limit = 0
    timeout = "2s"

  [[services.http_checks]]
    grace_period = "30s"
    interval = "10s"
    method = "get"
    path = "/health"
    protocol = "http"
    timeout = "2s"
"#
    );

    let fly_path = output.join("fly.toml");
    std::fs::write(&fly_path, fly_toml)?;
    println!("{} Generated: {}", "✓".bright_green(), fly_path.display());

    println!();
    println!("{}", "Deploy:".bright_yellow());
    println!("  fly launch");
    println!("  fly deploy");

    Ok(())
}

fn cmd_deploy_cloudflare(model: &str, output: &Path, name: Option<&str>) -> anyhow::Result<()> {
    println!(
        "{}",
        "☁️ Generating Cloudflare Workers Deployment"
            .bright_cyan()
            .bold()
    );
    println!();
    println!("{} Model: {}", "•".bright_blue(), model.cyan());
    println!("{} Output: {}", "•".bright_blue(), output.display());
    println!(
        "{} Worker name: {}",
        "•".bright_blue(),
        name.unwrap_or("realizar-worker").cyan()
    );
    println!();

    let worker_name = name.unwrap_or("realizar-worker");
    let wrangler_toml = format!(
        r#"# wrangler.toml - Cloudflare Workers configuration
# Generated by batuta deploy cloudflare

name = "{worker_name}"
main = "src/index.js"
compatibility_date = "2024-01-01"

[vars]
MODEL_REF = "{model}"

# Note: Cloudflare Workers have limited compute resources
# Consider using Cloudflare Pages with Functions for larger models
# or Cloudflare Workers with Durable Objects for persistent state
"#
    );

    let wrangler_path = output.join("wrangler.toml");
    std::fs::write(&wrangler_path, wrangler_toml)?;
    println!(
        "{} Generated: {}",
        "✓".bright_green(),
        wrangler_path.display()
    );

    println!();
    println!("{}", "Note:".bright_yellow());
    println!("  Cloudflare Workers have limited compute resources.");
    println!("  For ML inference, consider using:");
    println!("  - Cloudflare Workers AI (built-in LLM support)");
    println!("  - Edge proxy to Realizar server");
    println!();
    println!("{}", "Deploy:".bright_yellow());
    println!("  wrangler deploy");

    Ok(())
}
