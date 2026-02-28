//! Agent command implementations.
//!
//! CLI handlers for `batuta agent run|chat|validate|sign|verify-sig`.
//! Feature-gated behind `agents` (via `cli/mod.rs`).

use std::path::PathBuf;
use std::sync::Arc;

use crate::ansi_colors::Colorize;

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
    },

    /// Start an interactive chat session with an agent.
    Chat {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,
    },

    /// Validate an agent manifest without running it.
    ///
    /// Checks manifest syntax, consistency, and optionally validates
    /// the model file (BLAKE3 integrity, format detection).
    Validate {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Also validate the model file (G0: integrity, G1: format).
        #[arg(long)]
        check_model: bool,
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
        } => cmd_agent_run(&manifest, &prompt, max_iterations, daemon),
        AgentCommand::Chat { manifest } => cmd_agent_chat(&manifest),
        AgentCommand::Validate { manifest, check_model } => {
            cmd_agent_validate(&manifest, check_model)
        }
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

/// Build the LLM driver from manifest configuration.
fn build_driver(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<Box<dyn batuta::agent::driver::LlmDriver>> {
    // Resolve model path (supports model_path + model_repo)
    let resolved_path = manifest.model.resolve_model_path();

    // Phase 2: If resolved path exists, use RealizarDriver
    #[cfg(feature = "inference")]
    if let Some(ref model_path) = resolved_path {
        let driver =
            batuta::agent::driver::realizar::RealizarDriver::new(
                model_path.clone(),
                manifest.model.context_window,
            )
            .map_err(|e| anyhow::anyhow!("driver init: {e}"))?;
        return Ok(Box::new(driver));
    }

    // Fallback: MockDriver for dry-run / no model configured
    if resolved_path.is_some() {
        #[cfg(not(feature = "inference"))]
        {
            println!(
                "{} inference feature not enabled; using dry-run mode",
                "⚠".bright_yellow()
            );
            println!(
                "  Rebuild with: {}",
                "cargo build --features inference".cyan()
            );
        }
    } else {
        println!(
            "{} No model configured; using dry-run mode",
            "ℹ".bright_blue()
        );
        println!(
            "  Set model_path or model_repo in manifest",
        );
    }

    let driver = batuta::agent::driver::mock::MockDriver::single_response(
        "Hello! I'm running in dry-run mode (no local model configured). \
         Set model_path in your agent manifest to use local inference.",
    );
    Ok(Box::new(driver))
}

/// Build tool registry from manifest capabilities.
fn build_tool_registry(
    manifest: &batuta::agent::AgentManifest,
) -> batuta::agent::tool::ToolRegistry {
    use batuta::agent::capability::Capability;
    use batuta::agent::tool::ToolRegistry;

    let mut registry = ToolRegistry::new();

    for cap in &manifest.capabilities {
        match cap {
            Capability::Memory => {
                let memory = Arc::new(
                    batuta::agent::memory::InMemorySubstrate::new(),
                );
                registry.register(Box::new(
                    batuta::agent::tool::memory::MemoryTool::new(
                        memory,
                        manifest.name.clone(),
                    ),
                ));
            }
            Capability::Compute => {
                let cwd = std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                registry.register(Box::new(
                    batuta::agent::tool::compute::ComputeTool::new(cwd),
                ));
            }
            Capability::Shell { allowed_commands } => {
                let cwd = std::env::current_dir()
                    .unwrap_or_default();
                registry.register(Box::new(
                    batuta::agent::tool::shell::ShellTool::new(
                        allowed_commands.clone(),
                        cwd,
                    ),
                ));
            }
            // RAG and Browser tools require runtime-constructed oracles;
            // Network, Inference, Mcp are wired in Phases 3-4.
            _ => {}
        }
    }

    registry
}

/// Build memory substrate (in-memory for Phase 1, TruenoMemory
/// when rag feature is available).
fn build_memory() -> Box<dyn batuta::agent::memory::MemorySubstrate> {
    #[cfg(feature = "rag")]
    {
        match batuta::agent::memory::TruenoMemory::open_in_memory() {
            Ok(mem) => return Box::new(mem),
            Err(e) => {
                eprintln!(
                    "Warning: TruenoMemory init failed ({e}), \
                     falling back to InMemorySubstrate"
                );
            }
        }
    }
    Box::new(batuta::agent::memory::InMemorySubstrate::new())
}

/// Print a stream event to stdout.
fn print_stream_event(
    event: &batuta::agent::driver::StreamEvent,
) {
    use batuta::agent::driver::StreamEvent;
    match event {
        StreamEvent::PhaseChange { phase } => {
            println!(
                "  {} Phase: {phase:?}",
                "→".bright_blue()
            );
        }
        StreamEvent::ToolUseStart { name, .. } => {
            println!(
                "  {} Tool: {}",
                "⚙".bright_yellow(),
                name.cyan()
            );
        }
        StreamEvent::ToolUseEnd { name, result, .. } => {
            let preview = if result.len() > 80 {
                format!("{}...", &result[..77])
            } else {
                result.clone()
            };
            println!(
                "  {} {} → {}",
                "✓".green(),
                name,
                preview.dimmed()
            );
        }
        StreamEvent::TextDelta { text } => {
            print!("{text}");
        }
        StreamEvent::ContentComplete { .. } => {}
    }
}

/// Start an interactive chat session.
fn cmd_agent_chat(manifest_path: &PathBuf) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;
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

    Ok(())
}

/// Validate the model file (Jidoka: stop on defect).
///
/// G0: File exists and BLAKE3 hash computed (integrity).
/// G1: Format detection (GGUF magic, APR header, SafeTensors).
fn validate_model_file(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<()> {
    let model_path = manifest
        .model
        .resolve_model_path()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model_path or model_repo configured"
            )
        })?;

    println!();
    println!(
        "{} Model Validation (G0-G1)",
        "🔍".bright_cyan().bold()
    );
    println!("{}", "─".repeat(40).dimmed());

    // G0: File exists and integrity
    if !model_path.exists() {
        println!(
            "  {} G0 FAIL: model file not found: {}",
            "✗".bright_red(),
            model_path.display(),
        );
        anyhow::bail!(
            "G0: model file not found: {}",
            model_path.display()
        );
    }

    let metadata = std::fs::metadata(&model_path).map_err(|e| {
        anyhow::anyhow!("cannot stat {}: {e}", model_path.display())
    })?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

    // BLAKE3 hash for integrity
    let file_bytes = std::fs::read(&model_path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}",
            model_path.display()
        )
    })?;
    let hash = blake3::hash(&file_bytes);
    let hash_hex = hash.to_hex();

    println!(
        "  {} G0 PASS: {} ({:.1} MB)",
        "✓".green(),
        model_path.display(),
        size_mb,
    );
    println!(
        "    BLAKE3: {}",
        &hash_hex[..32],
    );

    // G1: Format detection via magic bytes
    let format = detect_model_format(&file_bytes);
    println!(
        "  {} G1: Format detected: {}",
        "✓".green(),
        format.bright_yellow(),
    );

    Ok(())
}

/// Detect model format from magic bytes.
fn detect_model_format(data: &[u8]) -> &'static str {
    if data.len() >= 4 {
        // GGUF magic: 0x46475547 ("GGUF" in LE)
        if data[..4] == [0x47, 0x47, 0x55, 0x46] {
            return "GGUF";
        }
        // APR v2 magic: "APR\x02"
        if data[..4] == [b'A', b'P', b'R', 0x02] {
            return "APR v2";
        }
        // SafeTensors: starts with JSON length (LE u64) then '{'
        if data.len() >= 9 && data[8] == b'{' {
            return "SafeTensors";
        }
    }
    "unknown"
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
                        .dimmed(),
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

/// Load and parse an agent manifest from TOML.
fn load_manifest(
    path: &PathBuf,
) -> anyhow::Result<batuta::agent::AgentManifest> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "Cannot read manifest {}: {e}",
            path.display()
        )
    })?;
    batuta::agent::AgentManifest::from_toml(&content).map_err(|e| {
        anyhow::anyhow!(
            "Invalid manifest {}: {e}",
            path.display()
        )
    })
}

/// Print a summary of the loaded manifest.
fn print_manifest_summary(
    manifest: &batuta::agent::AgentManifest,
) {
    println!(
        "{}",
        "🤖 Batuta Agent Runtime (Sovereign)"
            .bright_cyan()
            .bold()
    );
    println!("{}", "═".repeat(60).dimmed());
    println!(
        "{} Agent: {}",
        "•".bright_blue(),
        manifest.name.cyan()
    );
    println!(
        "{} Version: {}",
        "•".bright_blue(),
        manifest.version.dimmed()
    );
    println!(
        "{} Privacy: {:?}",
        "•".bright_blue(),
        manifest.privacy
    );
    println!(
        "{} Capabilities: {:?}",
        "•".bright_blue(),
        manifest.capabilities
    );
    println!(
        "{} Max iterations: {}",
        "•".bright_blue(),
        manifest.resources.max_iterations
    );

    if let Some(ref path) = manifest.model.model_path {
        println!(
            "{} Model: {}",
            "•".bright_blue(),
            path.display()
        );
    } else if let Some(ref repo) = manifest.model.model_repo {
        let quant = manifest
            .model
            .model_quantization
            .as_deref()
            .unwrap_or("q4_k_m");
        println!(
            "{} Model: {} ({})",
            "•".bright_blue(),
            repo.cyan(),
            quant,
        );
    } else {
        println!(
            "{} Model: {}",
            "•".bright_blue(),
            "none (specify model_path or model_repo)".dimmed()
        );
    }
}

/// Build a LoopGuard from manifest + optional override.
fn build_guard(
    manifest: &batuta::agent::AgentManifest,
    max_iterations: Option<u32>,
) -> (u32, u32) {
    let max_iter =
        max_iterations.unwrap_or(manifest.resources.max_iterations);
    let max_tools = manifest.resources.max_tool_calls;
    (max_iter, max_tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_valid_manifest() {
        let toml = r#"
name = "test-agent"
version = "0.1.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let manifest = load_manifest(&tmp.path().to_path_buf())
            .expect("should load");
        assert_eq!(manifest.name, "test-agent");
    }

    #[test]
    fn test_load_missing_file() {
        let result =
            load_manifest(&PathBuf::from("/nonexistent/agent.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_build_guard_with_override() {
        let manifest = batuta::agent::AgentManifest::default();
        let (max_iter, _) = build_guard(&manifest, Some(99));
        assert_eq!(max_iter, 99);
    }

    #[test]
    fn test_build_guard_from_manifest() {
        let mut manifest = batuta::agent::AgentManifest::default();
        manifest.resources.max_iterations = 42;
        let (max_iter, _) = build_guard(&manifest, None);
        assert_eq!(max_iter, 42);
    }

    #[test]
    fn test_validate_command_valid() {
        let toml = r#"
name = "valid"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_validate(&tmp.path().to_path_buf(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_with_needs_pull() {
        let toml = r#"
name = "repo-agent"
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
system_prompt = "hi"
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        // Should pass validation (warns about download)
        let result =
            cmd_agent_validate(&tmp.path().to_path_buf(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_check_model_no_path() {
        let toml = r#"
name = "no-model"
[model]
system_prompt = "hi"
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_validate(&tmp.path().to_path_buf(), true);
        // Should fail: no model configured
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_check_model_nonexistent() {
        let toml = r#"
name = "missing-model"
[model]
model_path = "/nonexistent/model.gguf"
system_prompt = "hi"
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_validate(&tmp.path().to_path_buf(), true);
        // Should fail: file not found (G0)
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_model_format() {
        // GGUF magic
        let gguf = [0x47u8, 0x47, 0x55, 0x46, 0, 0, 0, 0];
        assert_eq!(detect_model_format(&gguf), "GGUF");

        // APR v2
        let apr = [b'A', b'P', b'R', 0x02, 0, 0, 0, 0];
        assert_eq!(detect_model_format(&apr), "APR v2");

        // SafeTensors (8-byte length + '{')
        let st = [0u8, 0, 0, 0, 0, 0, 0, 0, b'{'];
        assert_eq!(detect_model_format(&st), "SafeTensors");

        // Unknown
        let unknown = [0u8; 4];
        assert_eq!(detect_model_format(&unknown), "unknown");
    }

    #[test]
    fn test_run_command_executes_loop() {
        let toml = r#"
name = "run-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        // No model_path → dry-run mode with MockDriver
        let result = cmd_agent_run(
            &tmp.path().to_path_buf(),
            "hello",
            None,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_driver_no_model_returns_mock() {
        let manifest = batuta::agent::AgentManifest::default();
        let driver = build_driver(&manifest);
        assert!(
            driver.is_ok(),
            "should return MockDriver when no model_path"
        );
    }

    #[test]
    fn test_build_tool_registry_memory() {
        use batuta::agent::capability::Capability;
        let mut manifest = batuta::agent::AgentManifest::default();
        manifest.capabilities = vec![Capability::Memory];
        let registry = build_tool_registry(&manifest);
        assert!(registry.get("memory").is_some());
    }

    #[test]
    fn test_build_tool_registry_compute() {
        use batuta::agent::capability::Capability;
        let mut manifest = batuta::agent::AgentManifest::default();
        manifest.capabilities = vec![Capability::Compute];
        let registry = build_tool_registry(&manifest);
        assert!(registry.get("compute").is_some());
    }

    #[test]
    fn test_build_tool_registry_shell() {
        use batuta::agent::capability::Capability;
        let mut manifest = batuta::agent::AgentManifest::default();
        manifest.capabilities = vec![Capability::Shell {
            allowed_commands: vec!["*".into()],
        }];
        let registry = build_tool_registry(&manifest);
        assert!(registry.get("shell").is_some());
    }

    #[test]
    fn test_build_memory_substrate() {
        let memory = build_memory();
        // Should not panic — returns either TruenoMemory or InMemory
        let _ = memory;
    }

    #[test]
    fn test_run_with_max_iterations_override() {
        let toml = r#"
name = "override-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result = cmd_agent_run(
            &tmp.path().to_path_buf(),
            "hello",
            Some(3),
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_command_invalid() {
        let toml = r#"
name = ""
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 0
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_validate(&tmp.path().to_path_buf(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_and_verify_roundtrip() {
        let toml = r#"
name = "sign-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");

        let sig_path = tmp.path().with_extension("toml.sig");
        let pk_path = sig_path.with_extension("pub");

        let result = cmd_agent_sign(
            &tmp.path().to_path_buf(),
            Some("tester"),
            Some(sig_path.clone()),
        );
        assert!(result.is_ok(), "sign failed: {result:?}");
        assert!(sig_path.exists(), "signature file not created");
        assert!(pk_path.exists(), "pubkey file not created");

        let result = cmd_agent_verify_sig(
            &tmp.path().to_path_buf(),
            Some(sig_path.clone()),
            &pk_path,
        );
        assert!(result.is_ok(), "verify failed: {result:?}");

        // Clean up
        let _ = std::fs::remove_file(&sig_path);
        let _ = std::fs::remove_file(&pk_path);
    }

    #[test]
    fn test_verify_fails_on_tampered() {
        let toml = r#"
name = "tamper-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");

        let sig_path = tmp.path().with_extension("toml.sig");
        let pk_path = sig_path.with_extension("pub");

        cmd_agent_sign(
            &tmp.path().to_path_buf(),
            None,
            Some(sig_path.clone()),
        )
        .expect("sign");

        // Tamper with manifest
        std::fs::write(tmp.path(), "name = \"tampered\"")
            .expect("tamper");

        let result = cmd_agent_verify_sig(
            &tmp.path().to_path_buf(),
            Some(sig_path.clone()),
            &pk_path,
        );
        assert!(result.is_err(), "should fail on tampered manifest");

        let _ = std::fs::remove_file(&sig_path);
        let _ = std::fs::remove_file(&pk_path);
    }

    #[test]
    fn test_contracts_command() {
        let result = cmd_agent_contracts();
        assert!(result.is_ok());
    }

    #[test]
    fn test_status_command() {
        let toml = r#"
name = "status-test"
version = "1.0.0"
privacy = "Sovereign"

[model]
max_tokens = 2048
temperature = 0.5
system_prompt = "hi"

[resources]
max_iterations = 15
max_tool_calls = 30
max_cost_usd = 1.50

[[capabilities]]
type = "rag"

[[capabilities]]
type = "memory"
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_status(&tmp.path().to_path_buf());
        assert!(result.is_ok());
    }

    #[test]
    fn test_status_command_no_capabilities() {
        let toml = r#"
name = "empty-caps"
[model]
system_prompt = "x"
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result =
            cmd_agent_status(&tmp.path().to_path_buf());
        assert!(result.is_ok());
    }

    #[test]
    fn test_pool_command_fan_out() {
        let toml = r#"
name = "pool-agent"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
        let tmp1 = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp1.path(), toml).expect("write");
        let tmp2 = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp2.path(), toml).expect("write");

        let manifests = vec![
            tmp1.path().to_path_buf(),
            tmp2.path().to_path_buf(),
        ];
        let result = cmd_agent_pool(&manifests, "hello", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pool_command_with_concurrency() {
        let toml = r#"
name = "pool-concurrent"
[model]
system_prompt = "hi"
[resources]
max_iterations = 3
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");

        let manifests = vec![tmp.path().to_path_buf()];
        let result =
            cmd_agent_pool(&manifests, "test", Some(1));
        assert!(result.is_ok());
    }
}
