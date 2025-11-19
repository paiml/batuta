mod analyzer;
mod config;
mod types;

use analyzer::analyze_project;
use clap::{Parser, Subcommand};
use colored::Colorize;
use config::BatutaConfig;
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

fn cmd_analyze(
    path: PathBuf,
    tdg: bool,
    languages: bool,
    dependencies: bool,
) -> anyhow::Result<()> {
    println!("{}", "🔍 Analyzing project...".bright_cyan().bold());
    println!();

    let analysis = analyze_project(&path, tdg, languages, dependencies)?;

    // Display results
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
    if !analysis.languages.is_empty() {
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

    // Dependencies
    if !analysis.dependencies.is_empty() {
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
                "ML frameworks detected - consider Aprender/Realizar for ML code"
                    .bright_yellow()
            );
            println!();
        }
    }

    // TDG Score
    if let Some(score) = analysis.tdg_score {
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
        println!(
            "  {} TDG Score: {}/100 ({})",
            "•".bright_blue(),
            format!("{:.1}", score).cyan(),
            grade
        );
        println!();
    }

    // Migration suggestions
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!(
        "  {} Run {} to convert project to Rust",
        "1.".bright_blue(),
        "batuta transpile".cyan()
    );
    println!(
        "  {} Run {} for performance optimization",
        "2.".bright_blue(),
        "batuta optimize".cyan()
    );
    println!(
        "  {} Run {} to verify equivalence",
        "3.".bright_blue(),
        "batuta validate".cyan()
    );
    println!();

    Ok(())
}

fn cmd_transpile(
    incremental: bool,
    cache: bool,
    modules: Option<Vec<String>>,
    ruchy: bool,
    repl: bool,
) -> anyhow::Result<()> {
    println!("🔄 Transpiling code...");
    if incremental {
        println!("   - Incremental mode enabled");
    }
    if cache {
        println!("   - Caching enabled");
    }
    if let Some(mods) = modules {
        println!("   - Modules: {}", mods.join(", "));
    }
    if ruchy {
        println!("   - Target: Ruchy");
    }
    if repl {
        println!("   - REPL mode");
    }
    warn!("Not yet implemented - Phase 2 (BATUTA-006)");
    Ok(())
}

fn cmd_optimize(
    enable_gpu: bool,
    enable_simd: bool,
    profile: OptimizationProfile,
    gpu_threshold: usize,
) -> anyhow::Result<()> {
    println!("⚡ Optimizing code...");
    println!("   Profile: {:?}", profile);
    if enable_gpu {
        println!("   - GPU acceleration enabled (threshold: {})", gpu_threshold);
    }
    if enable_simd {
        println!("   - SIMD vectorization enabled");
    }
    warn!("Not yet implemented - Phase 2 (BATUTA-006)");
    Ok(())
}

fn cmd_validate(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) -> anyhow::Result<()> {
    println!("✅ Validating equivalence...");
    if trace_syscalls {
        println!("   - Tracing syscalls");
    }
    if diff_output {
        println!("   - Generating diff output");
    }
    if run_original_tests {
        println!("   - Running original test suite");
    }
    if benchmark {
        println!("   - Running benchmarks");
    }
    warn!("Not yet implemented - Phase 2 (BATUTA-006)");
    Ok(())
}

fn cmd_build(release: bool, target: Option<String>, wasm: bool) -> anyhow::Result<()> {
    println!("🔨 Building Rust project...");
    if release {
        println!("   - Release mode");
    }
    if let Some(t) = target {
        println!("   - Target: {}", t);
    }
    if wasm {
        println!("   - WebAssembly target");
    }
    warn!("Not yet implemented - Phase 2 (BATUTA-006)");
    Ok(())
}

fn cmd_report(output: PathBuf, format: ReportFormat) -> anyhow::Result<()> {
    println!("📊 Generating migration report...");
    println!("   Output: {:?}", output);
    println!("   Format: {:?}", format);
    warn!("Not yet implemented - Phase 2 (BATUTA-006)");
    Ok(())
}
