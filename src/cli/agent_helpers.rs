//! Agent CLI helper functions.
//!
//! Build, validate, and format helpers shared across agent subcommands.

use std::path::PathBuf;
use std::sync::Arc;

use crate::ansi_colors::Colorize;

/// Auto-pull model if `model_repo` is set and file is missing.
///
/// Default timeout: 600 seconds (10 minutes) for large model downloads.
pub(super) fn try_auto_pull(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<()> {
    if let Some(repo) = manifest.model.needs_pull() {
        println!(
            "{} Auto-pulling model: {}",
            "⬇".bright_cyan(),
            repo.cyan(),
        );
        let quant = manifest
            .model
            .model_quantization
            .as_deref()
            .unwrap_or("q4_k_m");
        println!(
            "  Quantization: {}, Timeout: 600s",
            quant,
        );

        match manifest.model.auto_pull(600) {
            Ok(path) => {
                println!(
                    "{} Model downloaded: {}",
                    "✓".green(),
                    path.display(),
                );
            }
            Err(e) => {
                anyhow::bail!(
                    "auto-pull failed: {e}"
                );
            }
        }
    }
    Ok(())
}

/// Build the LLM driver from manifest configuration.
pub(super) fn build_driver(
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
pub(super) fn build_tool_registry(
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
pub(super) fn build_memory() -> Box<dyn batuta::agent::memory::MemorySubstrate> {
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
pub(super) fn print_stream_event(
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

/// Validate the model file (Jidoka: stop on defect).
///
/// G0: File exists and BLAKE3 hash computed (integrity).
/// G1: Format detection (GGUF magic, APR header, SafeTensors).
pub(super) fn validate_model_file(
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
pub(super) fn detect_model_format(data: &[u8]) -> &'static str {
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

/// Load and parse an agent manifest from TOML.
pub(super) fn load_manifest(
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
pub(super) fn print_manifest_summary(
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
pub(super) fn build_guard(
    manifest: &batuta::agent::AgentManifest,
    max_iterations: Option<u32>,
) -> (u32, u32) {
    let max_iter =
        max_iterations.unwrap_or(manifest.resources.max_iterations);
    let max_tools = manifest.resources.max_tool_calls;
    (max_iter, max_tools)
}
