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

use crate::ansi_colors::Colorize;
use batuta::agent::capability::Capability;
use batuta::agent::driver::LlmDriver;
use batuta::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use batuta::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use batuta::agent::tool::search::{GlobTool, GrepTool};
use batuta::agent::tool::shell::ShellTool;
use batuta::agent::tool::ToolRegistry;
use batuta::serve::backends::PrivacyTier;

/// Entry point for `batuta code`.
pub fn cmd_code(
    model: Option<PathBuf>,
    project: PathBuf,
    resume: Option<Option<String>>,
    prompt: Vec<String>,
    print: bool,
    max_turns: u32,
    manifest_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    // --project: change working directory for project instructions
    if project.as_os_str() != "." && project.is_dir() {
        std::env::set_current_dir(&project)?;
    }

    // Load manifest or build default
    let mut manifest = match manifest_path {
        Some(ref path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read manifest {}: {e}", path.display()))?;
            let m = batuta::agent::manifest::AgentManifest::from_toml(&content)
                .map_err(|e| anyhow::anyhow!("invalid manifest: {e}"))?;
            println!("{} Loaded manifest: {}", "✓".green(), path.display());
            m
        }
        None => build_default_manifest(true), // Always Sovereign
    };

    // --model flag overrides manifest model_path
    if let Some(ref model_path) = model {
        manifest.model.model_path = Some(model_path.clone());
    }

    // Build driver — Sovereign only, must have a local model
    let driver = super::agent::build_driver_pub(&manifest)?;

    // Contract: no_model_error — never silently use MockDriver
    if manifest.model.resolve_model_path().is_none() && manifest_path.is_none() {
        println!("{} No local model found. apr code requires a local model.\n", "✗".bright_red());
        println!("  Download a model (APR format preferred, GGUF also supported):");
        println!("    {} qwen2.5-coder:7b-q4_k_m", "apr pull".cyan());
        println!("    {} qwen3:8b-q4_k_m", "apr pull".cyan());
        println!();
        println!(
            "  Or place a .apr/.gguf file in {} (auto-discovered)",
            "~/.apr/models/".bright_yellow()
        );
        println!();
        println!("  Then run: {} or {} --model <path>", "batuta code".cyan(), "batuta code".cyan());
        std::process::exit(5);
    }

    // Build tool registry with coding tools
    let tools = build_code_tools(&manifest);

    // Build memory
    let memory = batuta::agent::memory::InMemorySubstrate::new();

    // Non-interactive mode: single prompt
    if print || !prompt.is_empty() {
        let prompt_text = if prompt.is_empty() {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        } else {
            prompt.join(" ")
        };
        return run_single_prompt(&manifest, driver.as_ref(), &tools, &memory, &prompt_text);
    }

    // --resume: load previous session
    let resume_session_id = match resume {
        Some(Some(id)) => Some(id), // --resume=<session-id>
        Some(None) => {
            // --resume (no ID): find most recent for cwd
            batuta::agent::session::SessionStore::find_recent_for_cwd().map(|m| m.id)
        }
        None => None,
    };

    // Interactive REPL (local inference is free — budget unlimited)
    batuta::agent::repl::run_repl(
        &manifest,
        driver.as_ref(),
        &tools,
        &memory,
        max_turns,
        f64::MAX,
        resume_session_id.as_deref(),
    )
}

/// Load project-level instructions from APR.md or CLAUDE.md.
///
/// Discovery order (per apr-code.md §3.5):
/// 1. `APR.md` in project root (preferred, stack-native)
/// 2. `CLAUDE.md` in project root (compatible with Claude Code)
///
/// Returns None if neither file exists. Truncates to 4KB to avoid
/// blowing up the context window on large instruction files.
fn load_project_instructions() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;

    // APR.md first (stack-native), then CLAUDE.md (compatibility)
    for filename in &["APR.md", "CLAUDE.md"] {
        let path = cwd.join(filename);
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let truncated = if content.len() > 4096 {
                    format!("{}...\n(truncated from {} bytes)", &content[..4096], content.len())
                } else {
                    content
                };
                return Some(truncated);
            }
        }
    }
    None
}

/// Gather project context — git info, file stats, language.
///
/// Injected into system prompt so the local model understands the
/// project it's working on (spec §6.2).
fn gather_project_context() -> String {
    let mut ctx = String::new();
    let cwd = std::env::current_dir().unwrap_or_default();
    ctx.push_str(&format!("Working directory: {}\n", cwd.display()));

    // Git info
    if let Ok(output) =
        std::process::Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]).output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ctx.push_str(&format!("Git branch: {branch}\n"));
        }
    }
    if let Ok(output) =
        std::process::Command::new("git").args(["diff", "--stat", "--no-color"]).output()
    {
        if output.status.success() {
            let diff = String::from_utf8_lossy(&output.stdout);
            let dirty_count = diff.lines().count().saturating_sub(1);
            if dirty_count > 0 {
                ctx.push_str(&format!("Dirty files: {dirty_count}\n"));
            }
        }
    }

    // Language detection via file extensions
    let mut rs_count = 0u32;
    let mut py_count = 0u32;
    let mut total = 0u32;
    if let Ok(entries) = std::fs::read_dir("src") {
        for e in entries.flatten() {
            total += 1;
            if let Some(ext) = e.path().extension() {
                match ext.to_str() {
                    Some("rs") => rs_count += 1,
                    Some("py") => py_count += 1,
                    _ => {}
                }
            }
        }
    }
    let lang = if rs_count > py_count {
        "Rust"
    } else if py_count > 0 {
        "Python"
    } else {
        "unknown"
    };
    ctx.push_str(&format!("Language: {lang} ({total} files in src/)\n"));

    // Cargo.toml presence
    if PathBuf::from("Cargo.toml").exists() {
        ctx.push_str("Build system: Cargo (Rust)\n");
    } else if PathBuf::from("pyproject.toml").exists() {
        ctx.push_str("Build system: pyproject.toml (Python)\n");
    }

    ctx
}

/// Build a default `AgentManifest` for coding tasks.
///
/// apr code is always Sovereign — all inference is local via realizar.
/// The `_offline` parameter is kept for API compatibility but ignored;
/// apr code never uses remote providers.
fn build_default_manifest(_offline: bool) -> AgentManifest {
    // Load project instructions and context
    let project_instructions = load_project_instructions();
    let project_context = gather_project_context();

    let mut system_prompt = CODE_SYSTEM_PROMPT.to_string();
    system_prompt.push_str(&format!("\n\n## Project Context\n\n{project_context}"));
    if let Some(ref instructions) = project_instructions {
        system_prompt.push_str(&format!("\n## Project Instructions\n\n{instructions}"));
    }

    AgentManifest {
        name: "apr-code".to_string(),
        description: "Interactive AI coding assistant".to_string(),
        privacy: PrivacyTier::Sovereign, // Always Sovereign — spec §5.4
        model: ModelConfig {
            system_prompt,
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
    tools.register(Box::new(ShellTool::new(vec!["*".into()], cwd)));

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
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;

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
///
/// Tool definitions are injected separately by `build_enriched_system()`
/// in chat_template.rs — this prompt focuses on behavior guidelines.
/// The model also receives full JSON schemas for each tool.
const CODE_SYSTEM_PROMPT: &str = "\
You are apr code, a sovereign AI coding assistant. All inference runs locally \
on the user's hardware — no data ever leaves the machine.

You help with software engineering tasks by reading code, making edits, running \
commands, and searching for information. You have access to tools listed below.

## How to Use Tools

When you need to use a tool, emit a <tool_call> block:

<tool_call>
{\"name\": \"file_read\", \"input\": {\"path\": \"src/main.rs\"}}
</tool_call>

You will receive the result in a <tool_result> block. Analyze it, then either \
use another tool or respond to the user. You can make multiple tool calls in \
sequence within a single turn.

## Guidelines

- Read files before modifying them — understand existing code first
- Use file_edit for targeted changes, file_write only for new files
- Run tests after changes: shell with cargo test or make test
- Prefer APR format (.apr) over GGUF when both are available — APR is the \
native model format with faster loading and row-major layout
- Use glob to find files, grep to search content
- Explain what you're doing concisely
- If a task is unclear, ask for clarification before making changes
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_default_manifest_always_sovereign() {
        // apr code is always Sovereign regardless of offline parameter
        let m1 = build_default_manifest(false);
        assert_eq!(m1.name, "apr-code");
        assert_eq!(m1.privacy, PrivacyTier::Sovereign);
        assert!(!m1.capabilities.is_empty());

        let m2 = build_default_manifest(true);
        assert_eq!(m2.privacy, PrivacyTier::Sovereign);
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
    fn test_default_manifest_is_sovereign() {
        let m = build_default_manifest(true);
        assert_eq!(m.privacy, PrivacyTier::Sovereign);
        // apr code is always Sovereign — even with offline=false
        let m2 = build_default_manifest(false);
        assert_eq!(m2.privacy, PrivacyTier::Sovereign, "apr code must always be Sovereign");
    }

    #[test]
    fn test_code_system_prompt_not_empty() {
        assert!(CODE_SYSTEM_PROMPT.len() > 200);
        assert!(CODE_SYSTEM_PROMPT.contains("tool_call"));
        assert!(CODE_SYSTEM_PROMPT.contains("file_read"));
        assert!(CODE_SYSTEM_PROMPT.contains("file_edit"));
        assert!(CODE_SYSTEM_PROMPT.contains("shell"));
        assert!(CODE_SYSTEM_PROMPT.contains("APR"));
        assert!(CODE_SYSTEM_PROMPT.contains("sovereign"));
    }

    #[test]
    fn test_load_project_instructions_from_claude_md() {
        // We're running in the batuta project which has a CLAUDE.md
        let instructions = load_project_instructions();
        // batuta has a CLAUDE.md, so this should find it
        assert!(instructions.is_some(), "expected to find CLAUDE.md in project root");
        let text = instructions.unwrap();
        assert!(
            text.contains("batuta") || text.contains("Batuta") || text.contains("CLAUDE"),
            "CLAUDE.md should mention the project"
        );
    }

    #[test]
    fn test_manifest_includes_project_instructions() {
        let m = build_default_manifest(true);
        // Since batuta has CLAUDE.md, the system prompt should include it
        assert!(
            m.model.system_prompt.contains("Project Instructions")
                || m.model.system_prompt.contains("sovereign"),
            "system prompt should contain either project instructions or base prompt"
        );
    }

    #[test]
    fn test_gather_project_context_has_content() {
        let ctx = gather_project_context();
        assert!(ctx.contains("Working directory:"), "should have cwd");
        // Running in batuta repo — should detect Rust/Cargo
        assert!(
            ctx.contains("Rust") || ctx.contains("Cargo") || ctx.contains("Language:"),
            "should detect language or build system: {ctx}"
        );
    }

    #[test]
    fn test_manifest_includes_project_context() {
        let m = build_default_manifest(true);
        assert!(
            m.model.system_prompt.contains("Project Context"),
            "system prompt should contain project context section"
        );
        assert!(
            m.model.system_prompt.contains("Working directory:"),
            "context should include working directory"
        );
    }
}
