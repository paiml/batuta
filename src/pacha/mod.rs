//! Pacha Model Registry CLI Module
//!
//! Provides CLI commands for interacting with the Pacha model registry,
//! offering an "ollama-like" experience for model management.
//!
//! ## Commands
//!
//! - `pull` - Download a model from registry or HuggingFace
//! - `list` - List cached models
//! - `rm` - Remove a model from cache
//! - `show` - Show model information
//! - `run` - Run inference on a model (interactive)
//!
//! ## Examples
//!
//! ```bash
//! # Pull a model
//! batuta pacha pull llama3:8b
//!
//! # List cached models
//! batuta pacha list
//!
//! # Show model info
//! batuta pacha show llama3
//!
//! # Remove a model
//! batuta pacha rm llama3:8b
//! ```

pub mod crypto;

use crate::ansi_colors::Colorize; // DEP-REDUCE: replaced colored crate
use clap::Subcommand;
use std::io::{self, Write};

// ============================================================================
// PACHA-CLI-001: Command Definitions
// ============================================================================

/// Pacha model registry commands
#[derive(Subcommand, Debug, Clone)]
pub enum PachaCommand {
    /// Pull a model from registry or HuggingFace
    Pull {
        /// Model reference (e.g., llama3, llama3:8b, hf://meta-llama/Llama-3-8B)
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if cached
        #[arg(short, long)]
        force: bool,

        /// Specific quantization (e.g., q4_k_m, q8_0, f16)
        #[arg(short, long)]
        quant: Option<String>,
    },

    /// List cached models
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,

        /// Output format (table, json)
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Remove a model from cache
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model: String,

        /// Remove all versions of this model
        #[arg(short, long)]
        all: bool,

        /// Skip confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Show model information
    Show {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// Show full metadata (including tensors)
        #[arg(short, long)]
        full: bool,
    },

    /// Search for models
    Search {
        /// Search query
        #[arg(value_name = "QUERY")]
        query: String,

        /// Limit results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show available aliases
    Aliases {
        /// Filter by pattern
        #[arg(value_name = "PATTERN")]
        pattern: Option<String>,
    },

    /// Add a custom alias
    Alias {
        /// Alias name
        #[arg(value_name = "NAME")]
        name: String,

        /// Target URI
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Show cache statistics
    Stats,

    /// Clean up old/unused models
    Prune {
        /// Days since last access (default: 30)
        #[arg(short, long, default_value = "30")]
        days: u64,

        /// Dry run (show what would be removed)
        #[arg(short = 'n', long)]
        dry_run: bool,
    },

    /// Pin a model (prevent eviction)
    Pin {
        /// Model reference to pin
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Unpin a model
    Unpin {
        /// Model reference to unpin
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Run interactive chat with a model
    Run {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// System prompt
        #[arg(short, long)]
        system: Option<String>,

        /// Load parameters from Modelfile
        #[arg(short, long, value_name = "FILE")]
        modelfile: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Max tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Context window size
        #[arg(long, default_value = "4096")]
        context: usize,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate a signing key pair
    Keygen {
        /// Output path for private key (default: ~/.pacha/signing-key.pem)
        #[arg(short, long)]
        output: Option<String>,

        /// Signer identity (e.g., email)
        #[arg(short, long)]
        identity: Option<String>,

        /// Force overwrite existing key
        #[arg(short, long)]
        force: bool,
    },

    /// Sign a model file
    Sign {
        /// Model file or reference to sign
        #[arg(value_name = "MODEL")]
        model: String,

        /// Path to signing key (default: ~/.pacha/signing-key.pem)
        #[arg(short, long)]
        key: Option<String>,

        /// Output signature file (default: <model>.sig)
        #[arg(short, long)]
        output: Option<String>,

        /// Signer identity
        #[arg(short, long)]
        identity: Option<String>,
    },

    /// Verify a model signature
    Verify {
        /// Model file or reference to verify
        #[arg(value_name = "MODEL")]
        model: String,

        /// Signature file (default: <model>.sig)
        #[arg(short, long)]
        signature: Option<String>,

        /// Expected signer public key (hex)
        #[arg(short, long)]
        key: Option<String>,
    },

    /// Encrypt a model file for distribution
    Encrypt {
        /// Model file to encrypt
        #[arg(value_name = "MODEL")]
        model: String,

        /// Output file (default: <model>.enc)
        #[arg(short, long)]
        output: Option<String>,

        /// Read password from environment variable
        #[arg(long, value_name = "VAR")]
        password_env: Option<String>,
    },

    /// Decrypt an encrypted model file
    Decrypt {
        /// Encrypted model file
        #[arg(value_name = "FILE")]
        file: String,

        /// Output file (default: original name without .enc)
        #[arg(short, long)]
        output: Option<String>,

        /// Read password from environment variable
        #[arg(long, value_name = "VAR")]
        password_env: Option<String>,
    },
}

// ============================================================================
// PACHA-CLI-002: Command Handler
// ============================================================================

/// Execute a pacha command
pub fn cmd_pacha(command: PachaCommand) -> anyhow::Result<()> {
    match command {
        PachaCommand::Pull {
            model,
            force,
            quant,
        } => cmd_pull(&model, force, quant.as_deref()),
        PachaCommand::List { verbose, format } => cmd_list(verbose, &format),
        PachaCommand::Rm { model, all, yes } => cmd_rm(&model, all, yes),
        PachaCommand::Show { model, full } => cmd_show(&model, full),
        PachaCommand::Search { query, limit } => cmd_search(&query, limit),
        PachaCommand::Aliases { pattern } => cmd_aliases(pattern.as_deref()),
        PachaCommand::Alias { name, target } => cmd_alias(&name, &target),
        PachaCommand::Stats => cmd_stats(),
        PachaCommand::Prune { days, dry_run } => cmd_prune(days, dry_run),
        PachaCommand::Pin { model } => cmd_pin(&model),
        PachaCommand::Unpin { model } => cmd_unpin(&model),
        PachaCommand::Run {
            model,
            system,
            modelfile,
            temperature,
            max_tokens,
            context,
            verbose,
        } => cmd_run(
            &model,
            system.as_deref(),
            modelfile.as_deref(),
            temperature,
            max_tokens,
            context,
            verbose,
        ),
        PachaCommand::Keygen {
            output,
            identity,
            force,
        } => crypto::cmd_keygen(output.as_deref(), identity.as_deref(), force),
        PachaCommand::Sign {
            model,
            key,
            output,
            identity,
        } => crypto::cmd_sign(
            &model,
            key.as_deref(),
            output.as_deref(),
            identity.as_deref(),
        ),
        PachaCommand::Verify {
            model,
            signature,
            key,
        } => crypto::cmd_verify(&model, signature.as_deref(), key.as_deref()),
        PachaCommand::Encrypt {
            model,
            output,
            password_env,
        } => crypto::cmd_encrypt(&model, output.as_deref(), password_env.as_deref()),
        PachaCommand::Decrypt {
            file,
            output,
            password_env,
        } => crypto::cmd_decrypt(&file, output.as_deref(), password_env.as_deref()),
    }
}

// ============================================================================
// PACHA-CLI-003: Pull Command
// ============================================================================

fn cmd_pull(model: &str, force: bool, quant: Option<&str>) -> anyhow::Result<()> {
    println!("{}", "⬇️  Pacha Model Pull".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Model:  {}", model.cyan());
    if let Some(q) = quant {
        println!("Quant:  {}", q.yellow());
    }
    if force {
        println!("Mode:   {}", "Force re-download".yellow());
    }
    println!();

    // Create progress bar callback
    let progress_callback = |downloaded: u64, total: u64, speed: f64| {
        let percent = if total > 0 {
            (downloaded as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let downloaded_mb = downloaded as f64 / (1024.0 * 1024.0);
        let total_mb = total as f64 / (1024.0 * 1024.0);
        let speed_mb = speed / (1024.0 * 1024.0);

        print!(
            "\r{} {:.1}% ({:.1}/{:.1} MB) @ {:.1} MB/s    ",
            create_progress_bar(percent, 30),
            percent,
            downloaded_mb,
            total_mb,
            speed_mb
        );
        io::stdout().flush().ok();
    };

    // Simulate pull (actual implementation would use pacha::fetcher)
    println!("{}", "Resolving model reference...".dimmed());

    // For now, show what would happen
    let resolved = resolve_model_ref(model, quant)?;
    println!("Resolved: {}", resolved.cyan());
    println!();

    // Check cache
    if !force && is_cached(&resolved) {
        println!("{} Model already cached", "✓".bright_green().bold());
        println!("  Use {} to re-download", "--force".yellow());
        return Ok(());
    }

    // Simulate download progress
    println!("{}", "Downloading...".dimmed());
    for i in 0..=100 {
        progress_callback(i * 40_000_000, 4_000_000_000, 50_000_000.0);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    println!();
    println!();

    println!(
        "{} Model downloaded successfully!",
        "✓".bright_green().bold()
    );
    println!();
    println!("Run with: {}", format!("batuta serve {}", model).cyan());

    Ok(())
}

// ============================================================================
// PACHA-CLI-004: List Command
// ============================================================================

fn cmd_list(verbose: bool, format: &str) -> anyhow::Result<()> {
    println!("{}", "📦 Cached Models".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Get cached models (simulation)
    let models = get_cached_models();

    if models.is_empty() {
        println!("{}", "No models cached.".dimmed());
        println!();
        println!("Pull a model with: {}", "batuta pacha pull llama3".cyan());
        return Ok(());
    }

    if format == "json" {
        // JSON output
        let json = serde_json::json!({
            "models": models.iter().map(|m| {
                serde_json::json!({
                    "name": m.0,
                    "size": m.1,
                    "modified": m.2
                })
            }).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        // Table output
        println!(
            "{:<30} {:>12} {:>20}",
            "NAME".dimmed(),
            "SIZE".dimmed(),
            "MODIFIED".dimmed()
        );
        println!("{}", "─".repeat(62).dimmed());

        for (name, size, modified) in &models {
            let size_str = format_size(*size);
            println!(
                "{:<30} {:>12} {:>20}",
                name.cyan(),
                size_str,
                modified.dimmed()
            );
        }
    }

    if verbose {
        println!();
        println!("{}", "─".repeat(62).dimmed());
        let total_size: u64 = models.iter().map(|m| m.1).sum();
        println!(
            "Total: {} models, {}",
            models.len().to_string().cyan(),
            format_size(total_size).yellow()
        );
    }

    Ok(())
}

// ============================================================================
// PACHA-CLI-005: Remove Command
// ============================================================================

fn cmd_rm(model: &str, all: bool, yes: bool) -> anyhow::Result<()> {
    println!("{}", "🗑️  Remove Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Model: {}", model.cyan());
    if all {
        println!("Mode:  {}", "Remove all versions".yellow());
    }
    println!();

    // Confirm unless -y flag
    if !yes {
        print!("Are you sure? [y/N] ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("{}", "Cancelled.".dimmed());
            return Ok(());
        }
    }

    // Simulate removal
    println!("{}", "Removing...".dimmed());
    std::thread::sleep(std::time::Duration::from_millis(200));

    println!(
        "{} Model removed: {}",
        "✓".bright_green().bold(),
        model.cyan()
    );

    Ok(())
}

// ============================================================================
// PACHA-CLI-006: Show Command
// ============================================================================

fn cmd_show(model: &str, full: bool) -> anyhow::Result<()> {
    println!("{}", "📋 Model Information".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Resolve and show info
    let resolved = resolve_model_ref(model, None)?;

    println!("Name:         {}", model.cyan());
    println!("Resolved:     {}", resolved);
    println!("Format:       {}", "GGUF".yellow());
    println!("Quantization: {}", "Q4_K_M".yellow());
    println!("Size:         {}", "4.37 GB".yellow());
    println!("Parameters:   {}", "8B".yellow());
    println!();

    println!("{}", "Architecture".bright_white().bold());
    println!("  Type:       LlamaForCausalLM");
    println!("  Context:    8192 tokens");
    println!("  Embedding:  4096");
    println!("  Layers:     32");
    println!("  Heads:      32");
    println!("  Vocab:      32000");

    if full {
        println!();
        println!("{}", "Tensors (first 10)".bright_white().bold());
        let tensors = [
            ("token_embd.weight", "[32000, 4096]", "Q4_K"),
            ("blk.0.attn_q.weight", "[4096, 4096]", "Q4_K"),
            ("blk.0.attn_k.weight", "[4096, 4096]", "Q4_K"),
            ("blk.0.attn_v.weight", "[4096, 4096]", "Q4_K"),
            ("blk.0.attn_output.weight", "[4096, 4096]", "Q4_K"),
        ];
        for (name, shape, quant) in &tensors {
            println!("  {} {} {}", name, shape.dimmed(), quant.dimmed());
        }
        println!("  ...");
    }

    Ok(())
}

// ============================================================================
// PACHA-CLI-007: Search Command
// ============================================================================

fn cmd_search(query: &str, limit: usize) -> anyhow::Result<()> {
    println!("{}", "🔍 Model Search".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Query: {}", query.cyan());
    println!("Limit: {}", limit);
    println!();

    // Simulate search results
    let results = [
        ("llama3", "Meta Llama 3 8B Instruct", "8B params"),
        ("llama3:70b", "Meta Llama 3 70B Instruct", "70B params"),
        ("mistral", "Mistral 7B Instruct v0.2", "7B params"),
        ("mixtral", "Mixtral 8x7B Instruct", "46.7B params"),
        ("phi3", "Microsoft Phi-3 Mini", "3.8B params"),
    ];

    let filtered: Vec<_> = results
        .iter()
        .filter(|(name, desc, _)| {
            name.to_lowercase().contains(&query.to_lowercase())
                || desc.to_lowercase().contains(&query.to_lowercase())
        })
        .take(limit)
        .collect();

    if filtered.is_empty() {
        println!("{}", "No results found.".dimmed());
        return Ok(());
    }

    println!(
        "{:<20} {:<35} {:>12}",
        "NAME".dimmed(),
        "DESCRIPTION".dimmed(),
        "SIZE".dimmed()
    );
    println!("{}", "─".repeat(70).dimmed());

    for (name, desc, size) in filtered {
        println!("{:<20} {:<35} {:>12}", name.cyan(), desc, size.dimmed());
    }

    println!();
    println!(
        "Pull with: {}",
        format!("batuta pacha pull {}", query).cyan()
    );

    Ok(())
}

// ============================================================================
// PACHA-CLI-008: Aliases Command
// ============================================================================

fn cmd_aliases(pattern: Option<&str>) -> anyhow::Result<()> {
    println!("{}", "🏷️  Model Aliases".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let aliases = [
        ("llama3", "hf://meta-llama/Meta-Llama-3-8B-Instruct"),
        ("llama3:70b", "hf://meta-llama/Meta-Llama-3-70B-Instruct"),
        ("llama3.1", "hf://meta-llama/Llama-3.1-8B-Instruct"),
        ("mistral", "hf://mistralai/Mistral-7B-Instruct-v0.2"),
        ("mixtral", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1"),
        ("phi3", "hf://microsoft/Phi-3-mini-4k-instruct"),
        ("gemma", "hf://google/gemma-7b-it"),
        ("qwen2", "hf://Qwen/Qwen2-7B-Instruct"),
        ("codellama", "hf://codellama/CodeLlama-7b-Instruct-hf"),
    ];

    let filtered: Vec<_> = if let Some(p) = pattern {
        aliases
            .iter()
            .filter(|(name, _)| name.contains(p))
            .collect()
    } else {
        aliases.iter().collect()
    };

    println!("{:<15} {}", "ALIAS".dimmed(), "TARGET".dimmed());
    println!("{}", "─".repeat(70).dimmed());

    for (alias, target) in filtered {
        println!("{:<15} {}", alias.cyan(), target);
    }

    println!();
    println!(
        "Add custom: {}",
        "batuta pacha alias mymodel hf://org/model".cyan()
    );

    Ok(())
}

// ============================================================================
// PACHA-CLI-009: Alias Command (Add)
// ============================================================================

fn cmd_alias(name: &str, target: &str) -> anyhow::Result<()> {
    println!("{}", "➕ Add Alias".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Alias:  {}", name.cyan());
    println!("Target: {}", target);
    println!();

    // Validate target
    if !target.starts_with("hf://")
        && !target.starts_with("pacha://")
        && !target.starts_with("file://")
    {
        println!(
            "{} Target should start with hf://, pacha://, or file://",
            "⚠".yellow()
        );
    }

    println!(
        "{} Alias added: {} → {}",
        "✓".bright_green().bold(),
        name.cyan(),
        target
    );

    Ok(())
}

// ============================================================================
// PACHA-CLI-010: Stats Command
// ============================================================================

fn cmd_stats() -> anyhow::Result<()> {
    println!("{}", "📊 Cache Statistics".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Simulated stats
    println!("{}", "Storage".bright_white().bold());
    println!("  Total Size:     {}", "23.5 GB".yellow());
    println!("  Max Size:       {}", "50.0 GB".dimmed());
    println!("  Usage:          {}", "47%".yellow());
    println!("  Available:      {}", "26.5 GB".dimmed());
    println!();

    println!("{}", "Models".bright_white().bold());
    println!("  Cached:         {}", "5".cyan());
    println!("  Pinned:         {}", "2".cyan());
    println!();

    println!("{}", "Performance".bright_white().bold());
    println!("  Cache Hits:     {}", "127".cyan());
    println!("  Cache Misses:   {}", "12".dimmed());
    println!("  Hit Rate:       {}", "91.4%".bright_green());
    println!();

    println!("{}", "Age".bright_white().bold());
    println!("  Oldest Entry:   {}", "14 days ago".dimmed());
    println!("  Most Accessed:  {}", "llama3:8b".cyan());

    Ok(())
}

// ============================================================================
// PACHA-CLI-011: Prune Command
// ============================================================================

fn cmd_prune(days: u64, dry_run: bool) -> anyhow::Result<()> {
    println!("{}", "🧹 Prune Cache".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Max Age:  {} days", days);
    if dry_run {
        println!("Mode:     {}", "Dry run (no changes)".yellow());
    }
    println!();

    // Simulated entries to prune
    let to_prune = [
        ("mistral:7b-q4", "2.1 GB", "45 days ago"),
        ("phi2", "1.8 GB", "38 days ago"),
    ];

    if to_prune.is_empty() {
        println!("{}", "No models to prune.".dimmed());
        return Ok(());
    }

    println!(
        "{:<25} {:>10} {:>15}",
        "MODEL".dimmed(),
        "SIZE".dimmed(),
        "LAST ACCESS".dimmed()
    );
    println!("{}", "─".repeat(50).dimmed());

    let mut total_size = 0u64;
    for (name, size, last_access) in &to_prune {
        println!(
            "{:<25} {:>10} {:>15}",
            name.red(),
            size,
            last_access.dimmed()
        );
        // Parse size for total
        if size.ends_with(" GB") {
            if let Ok(gb) = size.trim_end_matches(" GB").parse::<f64>() {
                total_size += (gb * 1024.0 * 1024.0 * 1024.0) as u64;
            }
        }
    }

    println!();
    println!("Would free: {}", format_size(total_size).yellow());

    if !dry_run {
        println!();
        println!(
            "{} Pruned {} models",
            "✓".bright_green().bold(),
            to_prune.len()
        );
    }

    Ok(())
}

// ============================================================================
// PACHA-CLI-012: Pin/Unpin Commands
// ============================================================================

fn cmd_pin(model: &str) -> anyhow::Result<()> {
    println!("{}", "📌 Pin Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Model: {}", model.cyan());
    println!();
    println!(
        "{} Model pinned (won't be evicted)",
        "✓".bright_green().bold()
    );

    Ok(())
}

fn cmd_unpin(model: &str) -> anyhow::Result<()> {
    println!("{}", "📍 Unpin Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Model: {}", model.cyan());
    println!();
    println!("{} Model unpinned", "✓".bright_green().bold());

    Ok(())
}

// ============================================================================
// PACHA-CLI-013: Run Command (Interactive Chat)
// ============================================================================

fn cmd_run(
    model: &str,
    system: Option<&str>,
    modelfile: Option<&str>,
    temperature: f32,
    max_tokens: Option<usize>,
    context: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    use std::io::BufRead;

    // Load configuration from modelfile if provided
    let (effective_system, effective_temp, effective_max_tokens) = if let Some(mf_path) = modelfile
    {
        let content = std::fs::read_to_string(mf_path)?;
        let manifest = parse_simple_modelfile(&content)?;
        (
            manifest.system.or_else(|| system.map(String::from)),
            manifest.temperature.unwrap_or(temperature),
            manifest.max_tokens.or(max_tokens),
        )
    } else {
        (system.map(String::from), temperature, max_tokens)
    };

    // Print header
    println!("{}", "🦙 Interactive Chat".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    println!("Model:       {}", model.cyan());
    if let Some(ref sys) = effective_system {
        println!("System:      {}", truncate_str(sys, 50).dimmed());
    }
    println!("Temperature: {}", format!("{:.1}", effective_temp).yellow());
    println!("Context:     {} tokens", context);
    if let Some(max) = effective_max_tokens {
        println!("Max Tokens:  {}", max);
    }
    println!();

    if verbose {
        println!("{}", "Loading model...".dimmed());
        std::thread::sleep(std::time::Duration::from_millis(500));
        println!("{} Model loaded", "✓".bright_green());
        println!();
    }

    println!(
        "{}",
        "Type your message and press Enter. Commands:".dimmed()
    );
    println!("{}", "  /bye, /exit, /quit - Exit chat".dimmed());
    println!("{}", "  /clear             - Clear context".dimmed());
    println!("{}", "  /system <prompt>   - Change system prompt".dimmed());
    println!("{}", "  /temp <value>      - Change temperature".dimmed());
    println!("{}", "  /save <file>       - Save conversation".dimmed());
    println!();
    println!("{}", "─".repeat(60).dimmed());

    // Chat state
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut current_system = effective_system.clone();
    let mut current_temp = effective_temp;

    // Add system message if present
    if let Some(ref sys) = current_system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
        });
    }

    // Interactive loop
    let stdin = io::stdin();
    loop {
        // Print prompt
        print!("\n{} ", ">>>".bright_green().bold());
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                // EOF
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                println!("{} Input error: {}", "✗".red(), e);
                continue;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match handle_chat_command(input, &mut messages, &mut current_system, &mut current_temp)
            {
                ChatCommandResult::Continue => continue,
                ChatCommandResult::Exit => break,
                ChatCommandResult::Error(msg) => {
                    println!("{} {}", "⚠".yellow(), msg);
                    continue;
                }
            }
        }

        // Add user message
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Simulate response (in real implementation, would call inference)
        print!("\n{} ", "<<<".bright_cyan().bold());
        io::stdout().flush()?;

        // Simulated streaming response
        let response = generate_simulated_response(input, &messages);
        for chunk in response.chars() {
            print!("{}", chunk);
            io::stdout().flush()?;
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        println!();

        // Add assistant message
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: response,
        });

        // Check context size and truncate if needed
        let token_estimate: usize = messages.iter().map(|m| m.content.len() / 4).sum();
        if token_estimate > context {
            if verbose {
                println!(
                    "{}",
                    format!("[Context truncated: ~{} tokens]", token_estimate).dimmed()
                );
            }
            truncate_context(&mut messages, context, current_system.is_some());
        }
    }

    println!();
    println!("{} Chat ended. Goodbye!", "👋".bright_cyan());

    Ok(())
}

/// Chat message structure
#[derive(Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Result of handling a chat command
enum ChatCommandResult {
    Continue,
    Exit,
    Error(String),
}

/// Handle chat slash commands
/// Handle /clear command: reset messages, re-add system prompt if set.
fn chat_cmd_clear(
    messages: &mut Vec<ChatMessage>,
    current_system: &Option<String>,
) -> ChatCommandResult {
    messages.clear();
    if let Some(ref sys) = current_system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
        });
    }
    println!("{} Context cleared", "✓".bright_green());
    ChatCommandResult::Continue
}

/// Handle /system command: update system prompt.
fn chat_cmd_system(
    arg: Option<&str>,
    messages: &mut Vec<ChatMessage>,
    current_system: &mut Option<String>,
) -> ChatCommandResult {
    let Some(prompt) = arg else {
        return ChatCommandResult::Error("Usage: /system <prompt>".to_string());
    };
    *current_system = Some(prompt.to_string());
    if let Some(msg) = messages.iter_mut().find(|m| m.role == "system") {
        msg.content = prompt.to_string();
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: prompt.to_string(),
            },
        );
    }
    println!("{} System prompt updated", "✓".bright_green());
    ChatCommandResult::Continue
}

/// Handle /temp command: set temperature.
fn chat_cmd_temp(arg: Option<&str>, current_temp: &mut f32) -> ChatCommandResult {
    let Some(val) = arg else {
        return ChatCommandResult::Error("Usage: /temp <value>".to_string());
    };
    match val.parse::<f32>() {
        Ok(t) if (0.0..=2.0).contains(&t) => {
            *current_temp = t;
            println!("{} Temperature set to {:.1}", "✓".bright_green(), t);
            ChatCommandResult::Continue
        }
        _ => ChatCommandResult::Error("Temperature must be between 0.0 and 2.0".to_string()),
    }
}

/// Handle /save command: save conversation to file.
fn chat_cmd_save(arg: Option<&str>, messages: &[ChatMessage]) -> ChatCommandResult {
    let Some(path) = arg else {
        return ChatCommandResult::Error("Usage: /save <file>".to_string());
    };
    match save_conversation(messages, path) {
        Ok(_) => {
            println!("{} Conversation saved to {}", "✓".bright_green(), path);
            ChatCommandResult::Continue
        }
        Err(e) => ChatCommandResult::Error(format!("Failed to save: {}", e)),
    }
}

fn handle_chat_command(
    input: &str,
    messages: &mut Vec<ChatMessage>,
    current_system: &mut Option<String>,
    current_temp: &mut f32,
) -> ChatCommandResult {
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim());

    match cmd.as_str() {
        "/bye" | "/exit" | "/quit" => ChatCommandResult::Exit,
        "/clear" => chat_cmd_clear(messages, current_system),
        "/system" => chat_cmd_system(arg, messages, current_system),
        "/temp" => chat_cmd_temp(arg, current_temp),
        "/save" => chat_cmd_save(arg, messages),
        "/help" => {
            println!("{}", "Commands:".bright_white().bold());
            println!("  /bye, /exit, /quit - Exit chat");
            println!("  /clear             - Clear context");
            println!("  /system <prompt>   - Change system prompt");
            println!("  /temp <value>      - Change temperature");
            println!("  /save <file>       - Save conversation");
            println!("  /help              - Show this help");
            ChatCommandResult::Continue
        }
        _ => ChatCommandResult::Error(format!("Unknown command: {}", cmd)),
    }
}

/// Simple modelfile parser for run command
struct SimpleModelfile {
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
}

fn parse_simple_modelfile(content: &str) -> anyhow::Result<SimpleModelfile> {
    let mut result = SimpleModelfile {
        system: None,
        temperature: None,
        max_tokens: None,
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            continue;
        }

        match parts[0].to_uppercase().as_str() {
            "SYSTEM" => {
                result.system = Some(parts[1].to_string());
            }
            "PARAMETER" => {
                let param_parts: Vec<&str> = parts[1].splitn(2, char::is_whitespace).collect();
                if param_parts.len() == 2 {
                    match param_parts[0].to_lowercase().as_str() {
                        "temperature" => {
                            result.temperature = param_parts[1].parse().ok();
                        }
                        "max_tokens" | "num_predict" => {
                            result.max_tokens = param_parts[1].parse().ok();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(result)
}

/// Generate simulated response (placeholder for real inference)
fn generate_simulated_response(input: &str, _messages: &[ChatMessage]) -> String {
    // Simple pattern matching for demo purposes
    let input_lower = input.to_lowercase();

    if input_lower.contains("hello") || input_lower.contains("hi") {
        return "Hello! How can I help you today?".to_string();
    }

    if input_lower.contains("how are you") {
        return "I'm doing well, thank you for asking! I'm ready to assist you with any questions or tasks you might have.".to_string();
    }

    if input_lower.contains("what is") || input_lower.contains("explain") {
        return format!(
            "That's an interesting question about \"{}\"! Let me explain: This is a simulated response. In a real implementation, I would provide a detailed explanation based on my training data and the context of our conversation.",
            input.chars().take(30).collect::<String>()
        );
    }

    if input_lower.contains("code") || input_lower.contains("program") {
        return "Here's a simple example:\n\n```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```\n\nThis is a basic Rust program that prints a greeting. Would you like me to explain any part of it?".to_string();
    }

    // Default response
    format!(
        "I understand you're asking about \"{}\". This is a simulated response for demonstration purposes. In production, this would use the actual inference engine to generate contextually appropriate responses.",
        truncate_str(input, 40)
    )
}

/// Truncate context to fit within window size
fn truncate_context(messages: &mut Vec<ChatMessage>, max_tokens: usize, has_system: bool) {
    // Keep system message and recent messages
    let start_idx = if has_system { 1 } else { 0 };

    while messages.len() > start_idx + 2 {
        let token_estimate: usize = messages.iter().map(|m| m.content.len() / 4).sum();
        if token_estimate <= max_tokens {
            break;
        }
        // Remove oldest non-system message
        messages.remove(start_idx);
    }
}

/// Save conversation to file
fn save_conversation(messages: &[ChatMessage], path: &str) -> anyhow::Result<()> {
    let mut output = String::new();

    for msg in messages {
        output.push_str(&format!(
            "[{}]\n{}\n\n",
            msg.role.to_uppercase(),
            msg.content
        ));
    }

    std::fs::write(path, output)?;
    Ok(())
}

/// Truncate string with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Resolve a model reference to full URI
fn resolve_model_ref(model: &str, quant: Option<&str>) -> anyhow::Result<String> {
    // Check for known aliases
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
    ];

    for (alias, target) in &aliases {
        if model == *alias {
            let mut result = target.to_string();
            if let Some(q) = quant {
                result = format!("{}:{}", result, q.to_uppercase());
            }
            return Ok(result);
        }
    }

    // If already a full URI, return as-is
    if model.contains("://") {
        return Ok(model.to_string());
    }

    // Otherwise, assume pacha:// scheme
    Ok(format!("pacha://{}", model))
}

/// Check if a model is cached (simulation)
fn is_cached(_model: &str) -> bool {
    false // Always simulate not cached for now
}

/// Get cached models (simulation)
fn get_cached_models() -> Vec<(String, u64, String)> {
    vec![
        (
            "llama3:8b-q4_k_m".to_string(),
            4_690_000_000,
            "2 hours ago".to_string(),
        ),
        (
            "mistral:7b-q4_k_m".to_string(),
            4_110_000_000,
            "1 day ago".to_string(),
        ),
        (
            "phi3:mini-q4_k_m".to_string(),
            2_390_000_000,
            "3 days ago".to_string(),
        ),
    ]
}

/// Format size in human-readable form
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Create a progress bar string
fn create_progress_bar(percent: f64, width: usize) -> String {
    let filled = (percent / 100.0 * width as f64) as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "{}[{}{}]{}",
        "\x1b[36m", // Cyan
        "█".repeat(filled),
        "░".repeat(empty),
        "\x1b[0m" // Reset
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Command Parsing Tests
    // ========================================================================

    #[test]
    fn test_resolve_model_ref_alias() {
        let resolved = resolve_model_ref("llama3", None).unwrap();
        assert!(resolved.starts_with("hf://"));
        assert!(resolved.contains("Llama-3-8B"));
    }

    #[test]
    fn test_resolve_model_ref_with_quant() {
        let resolved = resolve_model_ref("llama3", Some("q8_0")).unwrap();
        assert!(resolved.contains("Q8_0"));
    }

    #[test]
    fn test_resolve_model_ref_full_uri() {
        let resolved = resolve_model_ref("hf://custom/model", None).unwrap();
        assert_eq!(resolved, "hf://custom/model");
    }

    #[test]
    fn test_resolve_model_ref_unknown() {
        let resolved = resolve_model_ref("custom-model", None).unwrap();
        assert_eq!(resolved, "pacha://custom-model");
    }

    // ========================================================================
    // Format Tests
    // ========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(2048), "2 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(5 * 1024 * 1024), "5.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(4 * 1024 * 1024 * 1024), "4.00 GB");
    }

    // ========================================================================
    // Progress Bar Tests
    // ========================================================================

    #[test]
    fn test_progress_bar_empty() {
        let bar = create_progress_bar(0.0, 10);
        assert!(bar.contains("░░░░░░░░░░"));
    }

    #[test]
    fn test_progress_bar_half() {
        let bar = create_progress_bar(50.0, 10);
        assert!(bar.contains("█████"));
        assert!(bar.contains("░░░░░"));
    }

    #[test]
    fn test_progress_bar_full() {
        let bar = create_progress_bar(100.0, 10);
        assert!(bar.contains("██████████"));
    }

    // ========================================================================
    // Helper Tests
    // ========================================================================

    #[test]
    fn test_get_cached_models() {
        let models = get_cached_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|(n, _, _)| n.contains("llama")));
    }

    #[test]
    fn test_is_cached() {
        // Currently always returns false
        assert!(!is_cached("llama3"));
    }

    // ========================================================================
    // Command Enum Tests
    // ========================================================================

    #[test]
    fn test_pacha_command_clone() {
        let cmd = PachaCommand::Pull {
            model: "llama3".to_string(),
            force: false,
            quant: None,
        };
        let cloned = cmd.clone();
        if let PachaCommand::Pull { model, .. } = cloned {
            assert_eq!(model, "llama3");
        } else {
            panic!("Clone failed");
        }
    }

    #[test]
    fn test_pacha_command_debug() {
        let cmd = PachaCommand::List {
            verbose: true,
            format: "json".to_string(),
        };
        let debug = format!("{:?}", cmd);
        assert!(debug.contains("List"));
        assert!(debug.contains("verbose"));
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_truncate_str_short() {
        let s = "hello";
        assert_eq!(truncate_str(s, 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        let s = "hello";
        assert_eq!(truncate_str(s, 5), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        let s = "hello world this is a long string";
        let result = truncate_str(s, 15);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 15);
    }

    #[test]
    fn test_parse_simple_modelfile_system() {
        let content = "FROM llama3\nSYSTEM You are helpful.";
        let mf = parse_simple_modelfile(content).unwrap();
        assert_eq!(mf.system, Some("You are helpful.".to_string()));
    }

    #[test]
    fn test_parse_simple_modelfile_temperature() {
        let content = "FROM llama3\nPARAMETER temperature 0.8";
        let mf = parse_simple_modelfile(content).unwrap();
        assert_eq!(mf.temperature, Some(0.8));
    }

    #[test]
    fn test_parse_simple_modelfile_max_tokens() {
        let content = "FROM llama3\nPARAMETER max_tokens 512";
        let mf = parse_simple_modelfile(content).unwrap();
        assert_eq!(mf.max_tokens, Some(512));
    }

    #[test]
    fn test_parse_simple_modelfile_num_predict() {
        let content = "FROM llama3\nPARAMETER num_predict 256";
        let mf = parse_simple_modelfile(content).unwrap();
        assert_eq!(mf.max_tokens, Some(256));
    }

    #[test]
    fn test_parse_simple_modelfile_empty() {
        let content = "";
        let mf = parse_simple_modelfile(content).unwrap();
        assert!(mf.system.is_none());
        assert!(mf.temperature.is_none());
    }

    #[test]
    fn test_parse_simple_modelfile_comments() {
        let content = "# comment\nFROM llama3\n# another comment";
        let mf = parse_simple_modelfile(content).unwrap();
        assert!(mf.system.is_none());
    }

    #[test]
    fn test_generate_simulated_response_hello() {
        let response = generate_simulated_response("hello", &[]);
        assert!(response.contains("Hello"));
    }

    #[test]
    fn test_generate_simulated_response_code() {
        let response = generate_simulated_response("show me some code", &[]);
        assert!(response.contains("```"));
    }

    #[test]
    fn test_generate_simulated_response_default() {
        let response = generate_simulated_response("random query", &[]);
        assert!(response.contains("simulated"));
    }

    #[test]
    fn test_truncate_context_small() {
        let mut messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "hello".to_string(),
            },
        ];
        truncate_context(&mut messages, 1000, false);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_truncate_context_with_system() {
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "x".repeat(500),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "y".repeat(500),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "z".repeat(500),
            },
        ];
        truncate_context(&mut messages, 100, true);
        // Should keep system message
        assert_eq!(messages[0].role, "system");
    }

    #[test]
    fn test_chat_message_clone() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
        };
        let cloned = msg.clone();
        assert_eq!(cloned.role, "user");
        assert_eq!(cloned.content, "test");
    }

    #[test]
    fn test_run_command_enum() {
        let cmd = PachaCommand::Run {
            model: "llama3".to_string(),
            system: Some("You are helpful".to_string()),
            modelfile: None,
            temperature: 0.7,
            max_tokens: Some(1024),
            context: 4096,
            verbose: false,
        };
        if let PachaCommand::Run {
            model, temperature, ..
        } = cmd
        {
            assert_eq!(model, "llama3");
            assert!((temperature - 0.7).abs() < 0.001);
        } else {
            panic!("Expected Run command");
        }
    }
}
