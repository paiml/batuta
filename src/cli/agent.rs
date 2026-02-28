//! Agent command implementations.
//!
//! CLI handlers for `batuta agent run|chat|validate`.
//! Feature-gated behind `agents` (via `cli/mod.rs`).

use std::path::PathBuf;

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
    let manifest = load_manifest(manifest_path)?;

    if let Some(max_iter) = max_iterations {
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

    // Build runtime components
    let guard = build_guard(&manifest, max_iterations);

    println!(
        "{} Agent loop configured: max {} iterations, {} tool calls",
        "✓".green(),
        guard.0,
        guard.1
    );
    println!();

    // Note: Full async execution requires tokio runtime.
    // Phase 1 provides the foundation; Phase 2 adds RealizarDriver.
    println!("{}", "─".repeat(60).dimmed());
    println!("{}", "Note:".bright_yellow());
    println!(
        "  Agent runtime requires a local model. Load one first:"
    );
    println!();
    println!("  {} llama3:8b", "apr pull".cyan());
    println!();
    println!(
        "  Then point manifest model_path to the cached .gguf file."
    );

    if daemon {
        println!();
        println!(
            "{} Daemon mode ready. Waiting for shutdown signal...",
            "⏳".bright_blue()
        );
        // Phase 2: Replace with actual tokio::signal::ctrl_c() loop
    }

    Ok(())
}

/// Start an interactive chat session (Phase 2).
fn cmd_agent_chat(manifest_path: &PathBuf) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;
    print_manifest_summary(&manifest);

    println!();
    println!(
        "{} Interactive chat mode is planned for Phase 2.",
        "ℹ".bright_blue()
    );
    println!(
        "  Use {} for single-turn execution.",
        "batuta agent run".cyan()
    );

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
    fn test_run_command_daemon_flag() {
        let toml = r#"
name = "daemon-test"
version = "1.0.0"
[model]
system_prompt = "You are a daemon."
[resources]
max_iterations = 10
"#;
        let tmp = NamedTempFile::new().expect("tmp file");
        std::fs::write(tmp.path(), toml).expect("write");
        let result = cmd_agent_run(
            &tmp.path().to_path_buf(),
            "test daemon",
            None,
            true,
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
