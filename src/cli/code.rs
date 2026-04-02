//! `batuta code` — interactive AI coding assistant.
//!
//! Wires `agent::repl::run_repl()` into a CLI subcommand with
//! sensible defaults for coding tasks. Builds a default
//! `AgentManifest` with file/search/shell tools pre-registered.
//!
//! This is the batuta-side entrypoint. The `apr code` subcommand
//! in `apr-cli` delegates to this via `batuta::cli::code::cmd_code`.
//!
//! See: docs/specifications/components/apr-code.md

use std::path::PathBuf;
use std::sync::Arc;

use batuta::agent::capability::Capability;
use batuta::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use batuta::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use batuta::agent::tool::search::{GlobTool, GrepTool};
use batuta::agent::tool::shell::ShellTool;
use batuta::agent::tool::ToolRegistry;
use crate::ansi_colors::Colorize;
use batuta::serve::backends::PrivacyTier;

/// Entry point for `batuta code`.
pub fn cmd_code(
    prompt: Vec<String>,
    print: bool,
    offline: bool,
    max_turns: u32,
    budget: f64,
    manifest_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    // Load manifest or build default
    let manifest = match manifest_path {
        Some(ref path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read manifest {}: {e}", path.display()))?;
            let m = batuta::agent::manifest::AgentManifest::from_toml(&content)
                .map_err(|e| anyhow::anyhow!("invalid manifest: {e}"))?;
            println!("{} Loaded manifest: {}", "✓".green(), path.display());
            m
        }
        None => build_default_manifest(offline),
    };

    // Build driver (reuses agent helper logic)
    let driver = super::agent::build_driver_pub(&manifest)?;

    // Build tool registry with coding tools
    let tools = build_code_tools(&manifest);

    // Build memory
    let memory = batuta::agent::memory::InMemorySubstrate::new();

    // Non-interactive mode: single prompt
    if print || !prompt.is_empty() {
        let prompt_text = if prompt.is_empty() {
            // Read from stdin for piped input
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        } else {
            prompt.join(" ")
        };

        return run_single_prompt(&manifest, driver.as_ref(), &tools, &memory, &prompt_text);
    }

    // Interactive REPL
    batuta::agent::repl::run_repl(&manifest, driver.as_ref(), &tools, &memory, max_turns, budget)
}

/// Build a default `AgentManifest` for coding tasks.
fn build_default_manifest(offline: bool) -> AgentManifest {
    let privacy = if offline { PrivacyTier::Sovereign } else { PrivacyTier::Standard };

    AgentManifest {
        name: "apr-code".to_string(),
        description: "Interactive AI coding assistant".to_string(),
        privacy,
        model: ModelConfig {
            system_prompt: CODE_SYSTEM_PROMPT.to_string(),
            max_tokens: 4096,
            temperature: 0.0,
            ..ModelConfig::default()
        },
        resources: ResourceQuota {
            max_iterations: 50,
            max_tool_calls: 200,
            max_cost_usd: 0.0,
            max_tokens_budget: None,
        },
        capabilities: vec![
            Capability::FileRead { allowed_paths: vec!["*".into()] },
            Capability::FileWrite { allowed_paths: vec!["*".into()] },
            Capability::Shell { allowed_commands: vec!["*".into()] },
            Capability::Memory,
        ],
        ..AgentManifest::default()
    }
}

/// Register all coding tools.
fn build_code_tools(manifest: &AgentManifest) -> ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(FileReadTool::new(vec!["*".into()])));
    tools.register(Box::new(FileWriteTool::new(vec!["*".into()])));
    tools.register(Box::new(FileEditTool::new(vec!["*".into()])));
    tools.register(Box::new(GlobTool::new(vec!["*".into()])));
    tools.register(Box::new(GrepTool::new(vec!["*".into()])));
    tools.register(Box::new(
        ShellTool::new(vec!["*".into()], cwd),
    ));

    // Register memory tool
    let memory_sub = Arc::new(batuta::agent::memory::InMemorySubstrate::new());
    tools.register(Box::new(batuta::agent::tool::memory::MemoryTool::new(
        memory_sub,
        manifest.name.clone(),
    )));

    // RAG tool registration deferred to Phase 3 (requires trueno-rag index)

    tools
}

/// Run a single prompt (non-interactive mode).
fn run_single_prompt(
    manifest: &AgentManifest,
    driver: &dyn batuta::agent::driver::LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn batuta::agent::memory::MemorySubstrate,
    prompt: &str,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let result = rt.block_on(batuta::agent::runtime::run_agent_loop(
        manifest, prompt, driver, tools, memory, None,
    ));

    match result {
        Ok(r) => {
            println!("{}", r.text);
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// System prompt for the coding assistant.
const CODE_SYSTEM_PROMPT: &str = "\
You are an AI coding assistant. You help users with software engineering tasks \
by reading code, making edits, running commands, and searching for relevant information.

Available tools:
- file_read: Read file contents with line numbers
- file_write: Create or overwrite files
- file_edit: Replace a unique string in a file
- glob: Find files matching a pattern
- grep: Search file contents
- shell: Execute shell commands

Guidelines:
- Read files before modifying them
- Make targeted edits, not full rewrites
- Run tests after changes to verify correctness
- Explain what you're doing and why
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_default_manifest_online() {
        let m = build_default_manifest(false);
        assert_eq!(m.name, "apr-code");
        assert_eq!(m.privacy, PrivacyTier::Standard);
        assert!(!m.capabilities.is_empty());
    }

    #[test]
    fn test_build_default_manifest_offline() {
        let m = build_default_manifest(true);
        assert_eq!(m.privacy, PrivacyTier::Sovereign);
    }

    #[test]
    fn test_build_code_tools_registers_all() {
        let m = build_default_manifest(false);
        let tools = build_code_tools(&m);
        assert!(tools.get("file_read").is_some(), "missing file_read");
        assert!(tools.get("file_write").is_some(), "missing file_write");
        assert!(tools.get("file_edit").is_some(), "missing file_edit");
        assert!(tools.get("glob").is_some(), "missing glob");
        assert!(tools.get("grep").is_some(), "missing grep");
        assert!(tools.get("shell").is_some(), "missing shell");
        assert!(tools.get("memory").is_some(), "missing memory");
        assert!(tools.len() >= 7, "expected >=7 tools, got {}", tools.len());
    }

    #[test]
    fn test_code_system_prompt_not_empty() {
        assert!(CODE_SYSTEM_PROMPT.len() > 100);
        assert!(CODE_SYSTEM_PROMPT.contains("file_read"));
        assert!(CODE_SYSTEM_PROMPT.contains("file_edit"));
        assert!(CODE_SYSTEM_PROMPT.contains("shell"));
    }
}
