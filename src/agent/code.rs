//! Public entry point for `apr code` / `batuta code`.
//!
//! This module provides the library-level API that both the `batuta` binary
//! and `apr-cli` use to launch the coding assistant. All logic lives here;
//! CLI wrappers are thin dispatchers.
//!
//! PMAT-162: Phase 6 — makes `cmd_code` accessible from the library crate
//! so `apr-cli` can call `batuta::agent::code::cmd_code()` directly.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::capability::Capability;
use crate::agent::driver::LlmDriver;
use crate::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use crate::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use crate::agent::tool::search::{GlobTool, GrepTool};
use crate::agent::tool::shell::ShellTool;
use crate::agent::tool::ToolRegistry;
use crate::serve::backends::PrivacyTier;

/// Entry point for `batuta code` / `apr code`.
///
/// This is the public library API — callable from both the batuta binary
/// and apr-cli (PMAT-162). Handles model discovery, driver selection,
/// tool registration, and REPL launch.
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
            let m = AgentManifest::from_toml(&content)
                .map_err(|e| anyhow::anyhow!("invalid manifest: {e}"))?;
            eprintln!("✓ Loaded manifest: {}", path.display());
            m
        }
        None => build_default_manifest(),
    };

    // --model flag overrides manifest model_path
    if let Some(ref model_path) = model {
        manifest.model.model_path = Some(model_path.clone());
    }

    // PMAT-150: discover model with Jidoka validation (broken APR → GGUF fallback)
    discover_and_set_model(&mut manifest);

    // Contract: no_model_error — never silently use MockDriver
    if manifest.model.resolve_model_path().is_none() && manifest_path.is_none() {
        print_no_model_error();
        std::process::exit(exit_code::NO_MODEL);
    }

    // PMAT-160: Try AprServeDriver first (apr serve has full CUDA/GPU).
    // Falls back to embedded RealizarDriver if `apr` binary not found.
    let driver: Box<dyn LlmDriver> = if let Some(model_path) = manifest.model.resolve_model_path() {
        match crate::agent::driver::apr_serve::AprServeDriver::launch(
            model_path,
            manifest.model.context_window,
        ) {
            Ok(d) => Box::new(d),
            Err(e) => {
                eprintln!("⚠ apr serve unavailable ({e}), using embedded inference");
                build_fallback_driver(&manifest)?
            }
        }
    } else {
        build_fallback_driver(&manifest)?
    };

    // Build tool registry with coding tools
    let tools = build_code_tools(&manifest);

    // Build memory
    let memory = crate::agent::memory::InMemorySubstrate::new();

    // Non-interactive mode: single prompt
    // PMAT-161: Return exit code instead of process::exit() so driver Drop
    // runs and kills the apr serve subprocess (no zombie processes).
    if print || !prompt.is_empty() {
        let prompt_text = if prompt.is_empty() {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        } else {
            prompt.join(" ")
        };
        let code = run_single_prompt(&manifest, driver.as_ref(), &tools, &memory, &prompt_text);
        drop(driver); // Kill apr serve subprocess before exit
        std::process::exit(code);
    }

    // --resume: load previous session
    // PMAT-165: auto-resume prompt when recent session exists (spec §6.3)
    let resume_session_id = match resume {
        Some(Some(id)) => Some(id), // --resume=<session-id>
        Some(None) => {
            // --resume (no ID): find most recent for cwd
            crate::agent::session::SessionStore::find_recent_for_cwd().map(|m| m.id)
        }
        None => {
            // No --resume flag: check for recent session and prompt
            offer_auto_resume()
        }
    };

    // Interactive REPL (local inference is free — budget unlimited)
    crate::agent::repl::run_repl(
        &manifest,
        driver.as_ref(),
        &tools,
        &memory,
        max_turns,
        f64::MAX,
        resume_session_id.as_deref(),
    )
}

/// PMAT-165: Offer to resume a recent session if one exists for this directory.
///
/// Checks for sessions modified within the last 24 hours. If found,
/// prompts the user with session info and [Y/n] choice. Returns
/// the session ID to resume, or None to start fresh.
fn offer_auto_resume() -> Option<String> {
    use crate::agent::session::SessionStore;

    let manifest = SessionStore::find_recent_for_cwd()?;

    // Compute age string from the created timestamp
    let age = manifest
        .created
        .parse::<chrono::DateTime<chrono::Utc>>()
        .ok()
        .map(|created| {
            let elapsed = chrono::Utc::now().signed_duration_since(created);
            if elapsed.num_hours() > 0 {
                format!("{}h ago", elapsed.num_hours())
            } else {
                format!("{}m ago", elapsed.num_minutes().max(1))
            }
        })
        .unwrap_or_else(|| "recently".to_string());

    eprintln!("  Found previous session ({age}, {} turns)", manifest.turns);
    eprint!("  Resume? [Y/n] ");

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return None;
    }
    let input = input.trim().to_lowercase();

    if input.is_empty() || input == "y" || input == "yes" {
        Some(manifest.id)
    } else {
        None
    }
}

/// Build fallback driver (embedded RealizarDriver) when AprServeDriver unavailable.
fn build_fallback_driver(manifest: &AgentManifest) -> anyhow::Result<Box<dyn LlmDriver>> {
    #[cfg(feature = "inference")]
    {
        if let Some(model_path) = manifest.model.resolve_model_path() {
            let driver = crate::agent::driver::realizar::RealizarDriver::new(
                model_path,
                manifest.model.context_window,
            )?;
            return Ok(Box::new(driver));
        }
    }
    let _ = manifest;
    // No model or no inference feature — return MockDriver
    Ok(Box::new(crate::agent::driver::mock::MockDriver::single_response(
        "Hello! I'm running in dry-run mode. \
         Set model_path in your agent manifest or install the `apr` binary.",
    )))
}

/// Auto-discover model if none explicitly set (APR preferred over GGUF).
fn discover_and_set_model(manifest: &mut AgentManifest) {
    if manifest.model.model_path.is_some() || manifest.model.model_repo.is_some() {
        return;
    }
    let Some(discovered) = ModelConfig::discover_model() else {
        return;
    };
    let ext = discovered.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "gguf" && check_invalid_apr_in_search_dirs() {
        eprintln!(
            "⚠ APR model found but invalid (missing tokenizer). Using GGUF fallback: {}",
            discovered.display()
        );
        eprintln!("  Re-convert with: apr convert <source>.gguf -o <output>.apr\n");
    }
    manifest.model.model_path = Some(discovered);
}

/// Print actionable error when no local model is available.
fn print_no_model_error() {
    eprintln!("✗ No local model found. apr code requires a local model.\n");
    if check_invalid_apr_in_search_dirs() {
        eprintln!("  ⚠ APR model(s) found but invalid (missing embedded tokenizer).");
        eprintln!("  Re-convert: apr convert <source>.gguf -o <output>.apr\n");
    }
    eprintln!("  Download a model (APR format preferred):");
    eprintln!("    apr pull qwen2.5-coder:1.5b-q4k   (default, fast)");
    eprintln!("    apr pull qwen2.5-coder:7b-q4k     (recommended for complex tasks)");
    eprintln!();
    eprintln!("  Or place a .apr/.gguf file in ~/.apr/models/ (auto-discovered)");
    eprintln!();
    eprintln!("  Then run: apr code or apr code --model <path>");
}

/// Check if any APR files in standard model search dirs are invalid.
fn check_invalid_apr_in_search_dirs() -> bool {
    for dir in &ModelConfig::model_search_dirs() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "apr")
                    && !crate::agent::driver::validate::is_valid_model_file(&path)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Load project-level instructions from APR.md or CLAUDE.md.
fn load_project_instructions(max_bytes: usize) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;

    for filename in &["APR.md", "CLAUDE.md"] {
        let path = cwd.join(filename);
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if max_bytes == 0 {
                    return None;
                }
                let truncated = if content.len() > max_bytes {
                    let end = content
                        .char_indices()
                        .take_while(|(i, _)| *i < max_bytes)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(max_bytes.min(content.len()));
                    format!("{}...\n(truncated from {} bytes)", &content[..end], content.len())
                } else {
                    content
                };
                return Some(truncated);
            }
        }
    }
    None
}

/// Compute instruction budget based on model context window.
fn instruction_budget(context_window: usize) -> usize {
    if context_window < 4096 {
        return 0;
    }
    let budget = context_window / 4;
    budget.min(4096)
}

/// Gather project context — git info, file stats, language.
fn gather_project_context() -> String {
    let mut ctx = String::new();
    let cwd = std::env::current_dir().unwrap_or_default();
    ctx.push_str(&format!("Working directory: {}\n", cwd.display()));

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

    if PathBuf::from("Cargo.toml").exists() {
        ctx.push_str("Build system: Cargo (Rust)\n");
    } else if PathBuf::from("pyproject.toml").exists() {
        ctx.push_str("Build system: pyproject.toml (Python)\n");
    }

    ctx
}

/// Build a default `AgentManifest` for coding tasks.
fn build_default_manifest() -> AgentManifest {
    let ctx_window = 4096_usize;
    let budget = instruction_budget(ctx_window);
    let project_instructions = load_project_instructions(budget);
    let project_context = gather_project_context();

    let mut system_prompt = CODE_SYSTEM_PROMPT.to_string();
    system_prompt.push_str(&format!("\n\n## Project Context\n\n{project_context}"));
    if let Some(ref instructions) = project_instructions {
        system_prompt.push_str(&format!("\n## Project Instructions\n\n{instructions}"));
    }

    AgentManifest {
        name: "apr-code".to_string(),
        description: "Interactive AI coding assistant".to_string(),
        privacy: PrivacyTier::Sovereign,
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
            Capability::Rag,
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

    let memory_sub = Arc::new(crate::agent::memory::InMemorySubstrate::new());
    tools.register(Box::new(crate::agent::tool::memory::MemoryTool::new(
        memory_sub,
        manifest.name.clone(),
    )));

    // PMAT-163: dedicated pmat_query tool
    tools.register(Box::new(crate::agent::tool::pmat_query::PmatQueryTool::new()));

    #[cfg(feature = "rag")]
    {
        let oracle = Arc::new(crate::oracle::rag::RagOracle::new());
        tools.register(Box::new(crate::agent::tool::rag::RagTool::new(oracle, 5)));
    }

    tools
}

/// Exit codes for non-interactive mode (spec §9.1).
pub mod exit_code {
    pub const SUCCESS: i32 = 0;
    pub const AGENT_ERROR: i32 = 1;
    pub const BUDGET_EXHAUSTED: i32 = 2;
    pub const MAX_TURNS: i32 = 3;
    pub const SANDBOX_VIOLATION: i32 = 4;
    pub const NO_MODEL: i32 = 5;
}

/// Run a single prompt (non-interactive mode).
fn run_single_prompt(
    manifest: &AgentManifest,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn crate::agent::memory::MemorySubstrate,
    prompt: &str,
) -> i32 {
    use crate::agent::result::AgentError;

    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: failed to create tokio runtime: {e}");
            return exit_code::AGENT_ERROR;
        }
    };

    let result = rt.block_on(crate::agent::runtime::run_agent_loop(
        manifest, prompt, driver, tools, memory, None,
    ));

    match result {
        Ok(r) => {
            println!("{}", r.text);
            exit_code::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            match &e {
                AgentError::CircuitBreak(_) => exit_code::BUDGET_EXHAUSTED,
                AgentError::MaxIterationsReached => exit_code::MAX_TURNS,
                AgentError::CapabilityDenied { .. } => exit_code::SANDBOX_VIOLATION,
                _ => exit_code::AGENT_ERROR,
            }
        }
    }
}

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;

/// System prompt for the coding assistant.
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
- Use pmat_query for code search (quality-annotated results), glob for files, grep for content
- Explain what you're doing concisely
- If a task is unclear, ask for clarification before making changes
";
