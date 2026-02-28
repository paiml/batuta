//! Agent command implementations.
//!
//! CLI handlers for `batuta agent run|chat|validate`.
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
    Validate {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,
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
        AgentCommand::Validate { manifest } => cmd_agent_validate(&manifest),
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
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for Ctrl+C");
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
    // Phase 2: If model_path is set, use RealizarDriver
    #[cfg(feature = "inference")]
    if let Some(ref model_path) = manifest.model.model_path {
        let driver =
            batuta::agent::driver::realizar::RealizarDriver::new(
                model_path.clone(),
                manifest.model.context_window,
            )
            .map_err(|e| anyhow::anyhow!("driver init: {e}"))?;
        return Ok(Box::new(driver));
    }

    // Fallback: MockDriver for dry-run / no model configured
    if manifest.model.model_path.is_some() {
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
            "{} No model_path in manifest; using dry-run mode",
            "ℹ".bright_blue()
        );
        println!(
            "  Set model_path in manifest or run: {}",
            "apr pull llama3:8b".cyan()
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

/// Validate an agent manifest.
fn cmd_agent_validate(
    manifest_path: &PathBuf,
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
    } else {
        println!(
            "{} Model: {}",
            "•".bright_blue(),
            "none (specify model_path in manifest)".dimmed()
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
            cmd_agent_validate(&tmp.path().to_path_buf());
        assert!(result.is_ok());
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
            cmd_agent_validate(&tmp.path().to_path_buf());
        assert!(result.is_err());
    }
}
