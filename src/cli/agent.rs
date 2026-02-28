//! Agent command implementations.
//!
//! CLI handlers for `batuta agent run|chat|validate|sign|verify-sig`.
//! Feature-gated behind `agents` (via `cli/mod.rs`).

#[path = "agent_helpers.rs"]
mod agent_helpers;

use std::path::PathBuf;
use std::sync::Arc;

use crate::ansi_colors::Colorize;

use agent_helpers::{
    build_driver, build_guard, build_memory, build_tool_registry,
    detect_model_format, load_manifest, print_manifest_summary,
    print_stream_event, try_auto_pull, validate_model_file,
    validate_model_g2,
};

/// Agent subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum AgentCommand {
    /// Run an agent with a single prompt (non-interactive).
    Run {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Prompt to send to the agent.
        #[arg(long)]
        prompt: String,

        /// Override max iterations from manifest.
        #[arg(long)]
        max_iterations: Option<u32>,

        /// Run as a long-lived daemon (for forjar deployments).
        #[arg(long)]
        daemon: bool,

        /// Auto-download model via `apr pull` if not cached.
        #[arg(long)]
        auto_pull: bool,
    },

    /// Start an interactive chat session with an agent.
    Chat {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Auto-download model via `apr pull` if not cached.
        #[arg(long)]
        auto_pull: bool,
    },

    /// Validate an agent manifest without running it.
    ///
    /// Checks manifest syntax, consistency, and optionally validates
    /// the model file (BLAKE3 integrity, format detection, inference).
    Validate {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Validate model file (G0: integrity, G1: format).
        #[arg(long)]
        check_model: bool,

        /// Run inference sanity check (G2: probe prompt, entropy).
        /// Requires --check-model. Uses the configured driver.
        #[arg(long)]
        check_inference: bool,
    },

    /// Sign an agent manifest with Ed25519 (via pacha).
    Sign {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Signer identity label (optional).
        #[arg(long)]
        signer: Option<String>,

        /// Path to write the signature sidecar file.
        /// Defaults to `<manifest>.sig`.
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Verify an agent manifest signature.
    VerifySig {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Path to the signature sidecar file.
        /// Defaults to `<manifest>.sig`.
        #[arg(long)]
        signature: Option<PathBuf>,

        /// Path to the public key file (hex-encoded).
        #[arg(long)]
        pubkey: PathBuf,
    },

    /// Verify contract invariant bindings against the test suite.
    Contracts,

    /// Show agent manifest status and configuration.
    Status {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,
    },

    /// Fan-out multiple agents and collect results (multi-agent pool).
    Pool {
        /// Paths to agent manifests (one per agent).
        #[arg(long, required = true, num_args = 1..)]
        manifest: Vec<PathBuf>,

        /// Prompt to send to each agent.
        #[arg(long)]
        prompt: String,

        /// Maximum concurrent agents (default: number of manifests).
        #[arg(long)]
        concurrency: Option<usize>,
    },
}

/// Dispatch an agent subcommand.
pub fn cmd_agent(command: AgentCommand) -> anyhow::Result<()> {
    match command {
        AgentCommand::Run {
            manifest,
            prompt,
            max_iterations,
            daemon,
            auto_pull,
        } => cmd_agent_run(&manifest, &prompt, max_iterations, daemon, auto_pull),
        AgentCommand::Chat { manifest, auto_pull } => {
            cmd_agent_chat(&manifest, auto_pull)
        }
        AgentCommand::Validate {
            manifest,
            check_model,
            check_inference,
        } => cmd_agent_validate(&manifest, check_model, check_inference),
        AgentCommand::Sign {
            manifest,
            signer,
            output,
        } => cmd_agent_sign(&manifest, signer.as_deref(), output),
        AgentCommand::VerifySig {
            manifest,
            signature,
            pubkey,
        } => cmd_agent_verify_sig(&manifest, signature, &pubkey),
        AgentCommand::Contracts => cmd_agent_contracts(),
        AgentCommand::Status { manifest } => cmd_agent_status(&manifest),
        AgentCommand::Pool {
            manifest,
            prompt,
            concurrency,
        } => cmd_agent_pool(&manifest, &prompt, concurrency),
    }
}

/// Run an agent with a single prompt.
fn cmd_agent_run(
    manifest_path: &PathBuf,
    prompt: &str,
    max_iterations: Option<u32>,
    daemon: bool,
    auto_pull: bool,
) -> anyhow::Result<()> {
    let mut manifest = load_manifest(manifest_path)?;

    if let Some(max_iter) = max_iterations {
        manifest.resources.max_iterations = max_iter;
        println!(
            "{} Overriding max_iterations: {}",
            "⚙".bright_blue(),
            max_iter
        );
    }

    // Auto-pull model if needed and --auto-pull flag is set
    if auto_pull {
        try_auto_pull(&manifest)?;
    }

    print_manifest_summary(&manifest);

    if daemon {
        println!(
            "{} Daemon mode: agent will run as background service",
            "⚙".bright_blue()
        );
        println!(
            "  Send {} to gracefully shut down.",
            "SIGTERM/SIGINT".bright_yellow()
        );
    }

    println!();
    println!("{} {}", "Prompt:".bright_yellow(), prompt);
    println!();

    let guard = build_guard(&manifest, None);
    println!(
        "{} Agent loop configured: max {} iterations, {} tool calls",
        "✓".green(),
        guard.0,
        guard.1
    );
    println!();

    // Build tokio runtime
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    // Build driver based on manifest model_path
    let driver = build_driver(&manifest)?;

    // Register tools based on manifest capabilities
    let tools = build_tool_registry(&manifest);

    // Memory substrate
    let memory = build_memory();

    // Stream events to stdout
    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<batuta::agent::driver::StreamEvent>(64);

    println!(
        "{} Starting agent loop...",
        "▶".bright_green()
    );
    println!("{}", "─".repeat(60).dimmed());
    println!();

    let result = rt.block_on(async {
        let loop_result = batuta::agent::runtime::run_agent_loop(
            &manifest,
            prompt,
            driver.as_ref(),
            &tools,
            memory.as_ref(),
            Some(tx),
        )
        .await;

        // Drain stream events
        while let Ok(event) = rx.try_recv() {
            print_stream_event(&event);
        }

        loop_result
    });

    println!();
    println!("{}", "─".repeat(60).dimmed());

    match result {
        Ok(result) => {
            println!();
            println!(
                "{} {}",
                "Response:".bright_green().bold(),
                result.text
            );
            println!();
            println!(
                "{} Iterations: {}, Tool calls: {}, Tokens: {}/{}",
                "✓".green(),
                result.iterations,
                result.tool_calls,
                result.usage.input_tokens,
                result.usage.output_tokens,
            );
        }
        Err(e) => {
            println!(
                "{} Agent error: {e}",
                "✗".bright_red()
            );
            anyhow::bail!("agent loop failed: {e}");
        }
    }

    if daemon {
        println!();
        println!(
            "{} Daemon mode: waiting for shutdown signal...",
            "⏳".bright_blue()
        );
        rt.block_on(async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                eprintln!("signal handler error: {e}");
            }
        });
        println!(
            "\n{} Shutting down gracefully.",
            "✓".green()
        );
    }

    Ok(())
}

/// Start an interactive chat session.
fn cmd_agent_chat(
    manifest_path: &PathBuf,
    auto_pull: bool,
) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;

    if auto_pull {
        try_auto_pull(&manifest)?;
    }

    print_manifest_summary(&manifest);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    let driver = build_driver(&manifest)?;
    let tools = build_tool_registry(&manifest);
    let memory = build_memory();

    println!();
    println!(
        "{} Interactive chat. Type {} or {} to exit.",
        "💬".bright_cyan(),
        "quit".bright_yellow(),
        "Ctrl+C".bright_yellow(),
    );
    println!("{}", "─".repeat(60).dimmed());

    let stdin = std::io::stdin();
    let mut line_buf = String::new();

    loop {
        print!("\n{} ", "You>".bright_green().bold());
        use std::io::Write;
        std::io::stdout().flush().ok();

        line_buf.clear();
        let bytes = stdin.read_line(&mut line_buf)
            .map_err(|e| anyhow::anyhow!("stdin: {e}"))?;

        // EOF (Ctrl+D)
        if bytes == 0 {
            println!("\n{} Goodbye.", "✓".green());
            break;
        }

        let input = line_buf.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!("{} Goodbye.", "✓".green());
            break;
        }

        let result = rt.block_on(batuta::agent::runtime::run_agent_loop(
            &manifest,
            input,
            driver.as_ref(),
            &tools,
            memory.as_ref(),
            None,
        ));

        match result {
            Ok(result) => {
                println!(
                    "\n{} {}",
                    "Agent>".bright_cyan().bold(),
                    result.text
                );
                println!(
                    "{}",
                    format!(
                        "  [iter={}, tools={}, tokens={}/{}]",
                        result.iterations,
                        result.tool_calls,
                        result.usage.input_tokens,
                        result.usage.output_tokens,
                    )
                    .dimmed()
                );
            }
            Err(e) => {
                println!(
                    "\n{} Error: {e}",
                    "✗".bright_red()
                );
            }
        }
    }

    Ok(())
}

/// Validate an agent manifest (and optionally the model file).
fn cmd_agent_validate(
    manifest_path: &PathBuf,
    check_model: bool,
    check_inference: bool,
) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;

    match manifest.validate() {
        Ok(()) => {
            println!(
                "{} Manifest is valid: {}",
                "✓".green(),
                manifest_path.display()
            );
            print_manifest_summary(&manifest);
        }
        Err(errors) => {
            println!(
                "{} Manifest validation failed:",
                "✗".bright_red()
            );
            for err in &errors {
                println!("  {} {}", "•".bright_red(), err);
            }
            anyhow::bail!(
                "{} validation error(s) in {}",
                errors.len(),
                manifest_path.display()
            );
        }
    }

    // Auto-pull check
    if let Some(repo) = manifest.model.needs_pull() {
        println!();
        println!(
            "{} Model needs download: {}",
            "⚠".bright_yellow(),
            repo,
        );
        if let Some(path) = manifest.model.resolve_model_path() {
            println!(
                "  Expected at: {}",
                path.display()
            );
        }
        println!(
            "  Run: {} {} {}",
            "apr pull".cyan(),
            repo,
            manifest
                .model
                .model_quantization
                .as_deref()
                .unwrap_or("q4_k_m"),
        );
    }

    if check_model {
        validate_model_file(&manifest)?;
    }

    if check_inference {
        validate_model_g2(&manifest)?;
    }

    Ok(())
}

/// Sign an agent manifest with Ed25519 (Jidoka: tamper detection).
fn cmd_agent_sign(
    manifest_path: &PathBuf,
    signer: Option<&str>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(manifest_path).map_err(|e| {
        anyhow::anyhow!(
            "Cannot read manifest {}: {e}",
            manifest_path.display()
        )
    })?;

    // Generate a new signing key (in production, load from secure store)
    let signing_key = pacha::signing::SigningKey::generate();
    let verifying_key = signing_key.verifying_key();

    let sig = batuta::agent::signing::sign_manifest(
        &content,
        &signing_key,
        signer,
    );

    let sig_path = output.unwrap_or_else(|| {
        let mut p = manifest_path.clone();
        let ext = p
            .extension()
            .map(|e| format!("{}.sig", e.to_string_lossy()))
            .unwrap_or_else(|| "sig".into());
        p.set_extension(ext);
        p
    });

    let sig_toml = batuta::agent::signing::signature_to_toml(&sig);
    std::fs::write(&sig_path, &sig_toml).map_err(|e| {
        anyhow::anyhow!(
            "Cannot write signature to {}: {e}",
            sig_path.display()
        )
    })?;

    // Write public key alongside
    let pk_path = sig_path.with_extension("pub");
    std::fs::write(&pk_path, verifying_key.to_hex()).map_err(|e| {
        anyhow::anyhow!(
            "Cannot write public key to {}: {e}",
            pk_path.display()
        )
    })?;

    println!(
        "{} Manifest signed: {}",
        "✓".green(),
        manifest_path.display()
    );
    println!(
        "  Signature: {}",
        sig_path.display()
    );
    println!(
        "  Public key: {}",
        pk_path.display()
    );
    println!(
        "  Hash: {}",
        &sig.content_hash[..16]
    );
    if let Some(ref s) = sig.signer {
        println!("  Signer: {s}");
    }

    Ok(())
}

/// Verify an agent manifest signature (Jidoka: stop on tampered).
fn cmd_agent_verify_sig(
    manifest_path: &PathBuf,
    signature_path: Option<PathBuf>,
    pubkey_path: &PathBuf,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(manifest_path).map_err(|e| {
        anyhow::anyhow!(
            "Cannot read manifest {}: {e}",
            manifest_path.display()
        )
    })?;

    let sig_path = signature_path.unwrap_or_else(|| {
        let mut p = manifest_path.clone();
        let ext = p
            .extension()
            .map(|e| format!("{}.sig", e.to_string_lossy()))
            .unwrap_or_else(|| "sig".into());
        p.set_extension(ext);
        p
    });

    let sig_content =
        std::fs::read_to_string(&sig_path).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read signature {}: {e}",
                sig_path.display()
            )
        })?;

    let sig = batuta::agent::signing::signature_from_toml(&sig_content)
        .map_err(|e| anyhow::anyhow!("Invalid signature: {e}"))?;

    let pk_hex =
        std::fs::read_to_string(pubkey_path).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read public key {}: {e}",
                pubkey_path.display()
            )
        })?;

    let vk =
        pacha::signing::VerifyingKey::from_hex(pk_hex.trim())
            .map_err(|e| {
                anyhow::anyhow!("Invalid public key: {e}")
            })?;

    match batuta::agent::signing::verify_manifest(&content, &sig, &vk)
    {
        Ok(()) => {
            println!(
                "{} Signature valid: {}",
                "✓".green(),
                manifest_path.display()
            );
            if let Some(ref signer) = sig.signer {
                println!("  Signer: {signer}");
            }
            println!(
                "  Hash: {}",
                &sig.content_hash[..16]
            );
            Ok(())
        }
        Err(e) => {
            println!(
                "{} Signature verification FAILED: {e}",
                "✗".bright_red()
            );
            anyhow::bail!(
                "manifest signature verification failed: {e}"
            );
        }
    }
}

/// Verify contract invariant bindings against the test suite.
fn cmd_agent_contracts() -> anyhow::Result<()> {
    let yaml = include_str!("../../contracts/agent-loop-v1.yaml");
    let contract = batuta::agent::contracts::parse_contract(yaml)
        .map_err(|e| anyhow::anyhow!("contract parse: {e}"))?;

    println!(
        "{} Contract: {} v{}",
        "📋".bright_blue(),
        contract.contract.name.cyan(),
        contract.contract.version,
    );
    println!(
        "  {}",
        contract.contract.description.dimmed()
    );
    println!();

    println!(
        "{} Invariants ({})",
        "•".bright_blue(),
        contract.invariants.len(),
    );
    for inv in &contract.invariants {
        println!(
            "  {} {} — {}",
            inv.id.bright_yellow(),
            inv.name,
            inv.description.dimmed(),
        );
        println!(
            "    Test: {}",
            inv.test_binding.cyan()
        );
    }
    println!();

    println!(
        "{} Verification targets:",
        "•".bright_blue(),
    );
    println!(
        "  Coverage: {}%  Mutation: {}%",
        contract.verification.coverage_target,
        contract.verification.mutation_target,
    );
    println!(
        "  Complexity: cyclomatic {} / cognitive {}",
        contract.verification.complexity_max_cyclomatic,
        contract.verification.complexity_max_cognitive,
    );
    println!();

    println!(
        "{} Run {} to check bindings.",
        "ℹ".bright_blue(),
        "cargo test --features agents -- test_all_contract_bindings_exist"
            .cyan(),
    );

    Ok(())
}

/// Display agent manifest status and configuration.
fn cmd_agent_status(
    manifest_path: &PathBuf,
) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;

    if let Err(errors) = manifest.validate() {
        println!(
            "{} Manifest has validation errors:",
            "⚠".bright_yellow(),
        );
        for e in &errors {
            println!("  {} {e}", "✗".red());
        }
        println!();
    }

    print_manifest_summary(&manifest);
    println!();

    // Resource quotas
    println!("{}", "Resource Quotas".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!(
        "  Max iterations:  {}",
        manifest.resources.max_iterations
    );
    println!(
        "  Max tool calls:  {}",
        manifest.resources.max_tool_calls
    );
    let budget = if manifest.resources.max_cost_usd > 0.0 {
        format!("${:.4}", manifest.resources.max_cost_usd)
    } else {
        "unlimited (sovereign)".to_string()
    };
    println!("  Cost budget:     {budget}");
    println!();

    // Model configuration
    println!("{}", "Model Configuration".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!(
        "  Max tokens:      {}",
        manifest.model.max_tokens
    );
    println!(
        "  Temperature:     {}",
        manifest.model.temperature
    );
    if let Some(ref ctx) = manifest.model.context_window {
        println!("  Context window:  {ctx}");
    } else {
        println!("  Context window:  auto-detect");
    }
    if let Some(ref repo) = manifest.model.model_repo {
        println!("  Model repo:      {repo}");
        let quant = manifest
            .model
            .model_quantization
            .as_deref()
            .unwrap_or("q4_k_m");
        println!("  Quantization:    {quant}");
        if let Some(resolved) = manifest.model.resolve_model_path() {
            let exists = if resolved.exists() { "found" } else { "not found" };
            println!("  Resolved path:   {} ({exists})", resolved.display());
        }
    }
    if let Some(ref remote) = manifest.model.remote_model {
        println!("  Remote model:    {remote}");
    }
    println!();

    // Capabilities
    println!("{}", "Capabilities".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    if manifest.capabilities.is_empty() {
        println!("  (none — deny-by-default)");
    }
    for cap in &manifest.capabilities {
        println!("  {} {cap:?}", "•".bright_blue());
    }
    println!();

    println!(
        "{} {}",
        "✓".green(),
        "Agent manifest loaded successfully.".green(),
    );

    Ok(())
}

/// Fan-out multiple agents and collect results.
fn cmd_agent_pool(
    manifest_paths: &[PathBuf],
    prompt: &str,
    concurrency: Option<usize>,
) -> anyhow::Result<()> {
    use batuta::agent::pool::{AgentPool, SpawnConfig};

    let manifests: Vec<batuta::agent::AgentManifest> = manifest_paths
        .iter()
        .map(|p| load_manifest(p))
        .collect::<Result<_, _>>()?;

    let max_concurrent =
        concurrency.unwrap_or(manifests.len());

    println!(
        "{} Multi-Agent Pool",
        "🔀".bright_cyan().bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!(
        "  Agents: {}  Concurrency: {}",
        manifests.len(),
        max_concurrent,
    );
    println!(
        "  Prompt: {}",
        prompt.bright_yellow()
    );
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    let driver = build_driver(
        manifests.first().unwrap_or(
            &batuta::agent::AgentManifest::default(),
        ),
    )?;
    let driver: Arc<dyn batuta::agent::driver::LlmDriver> =
        Arc::from(driver);
    let memory: Arc<dyn batuta::agent::memory::MemorySubstrate> =
        Arc::from(build_memory());

    let configs: Vec<SpawnConfig> = manifests
        .iter()
        .map(|m| SpawnConfig {
            manifest: m.clone(),
            query: prompt.to_string(),
        })
        .collect();

    let agent_count = configs.len();

    rt.block_on(async {
        let mut pool = AgentPool::new(
            driver, max_concurrent,
        )
        .with_memory(memory);

        println!(
            "{} Spawning {} agents...",
            "▶".bright_green(),
            agent_count,
        );

        let ids = pool.fan_out(configs).map_err(|e| {
            anyhow::anyhow!("pool fan-out: {e}")
        })?;
        for id in &ids {
            println!(
                "  {} Agent {id} spawned",
                "•".bright_blue(),
            );
        }
        println!();

        println!(
            "{} Waiting for results...",
            "⏳".bright_blue(),
        );
        let results = pool.join_all().await;

        println!();
        println!(
            "{} Results ({}/{})",
            "✓".green(),
            results.len(),
            ids.len(),
        );
        println!("{}", "─".repeat(60).dimmed());

        for (id, result) in &results {
            match result {
                Ok(r) => {
                    println!(
                        "  {} Agent {id}: {}",
                        "✓".green(),
                        r.text,
                    );
                    println!(
                        "    {}",
                        format!(
                            "[iter={}, tools={}, tokens={}/{}]",
                            r.iterations,
                            r.tool_calls,
                            r.usage.input_tokens,
                            r.usage.output_tokens,
                        )
                        .dimmed()
                    );
                }
                Err(e) => {
                    println!(
                        "  {} Agent {id}: {e}",
                        "✗".bright_red(),
                    );
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
