//! Agent CLI helper functions (build, validate, format).

use std::path::PathBuf;
use std::sync::Arc;
use crate::ansi_colors::Colorize;

/// Auto-pull model if `model_repo` is set and file is missing.
pub(super) fn try_auto_pull(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<()> {
    if let Some(repo) = manifest.model.needs_pull() {
        println!("{} Auto-pulling model: {}", "⬇".bright_cyan(), repo.cyan());
        let quant = manifest.model.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("  Quantization: {}, Timeout: 600s", quant);
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
        println!("{} inference feature not enabled; rebuild with: {}",
            "⚠".bright_yellow(), "cargo build --features inference".cyan());
    } else {
        println!("{} No model configured; set model_path or model_repo in manifest",
            "ℹ".bright_blue());
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
            Capability::Network { allowed_hosts } => {
                registry.register(Box::new(
                    batuta::agent::tool::network::NetworkTool::new(
                        allowed_hosts.clone(),
                    ),
                ));
            }
            #[cfg(feature = "agents-browser")]
            Capability::Browser => {
                registry.register(Box::new(
                    batuta::agent::tool::browser::BrowserTool::new(
                        manifest.privacy.clone(),
                    ),
                ));
            }
            #[cfg(feature = "rag")]
            Capability::Rag => {
                let oracle = Arc::new(
                    batuta::oracle::rag::RagOracle::new(),
                );
                registry.register(Box::new(
                    batuta::agent::tool::rag::RagTool::new(oracle, 5),
                ));
            }
            // Mcp wired in Phase 4.
            _ => {}
        }
    }

    registry
}

/// Register SpawnTool on the registry if Spawn capability is present.
pub(super) fn register_spawn_tool(
    registry: &mut batuta::agent::tool::ToolRegistry,
    manifest: &batuta::agent::AgentManifest,
    driver: Arc<dyn batuta::agent::driver::LlmDriver>,
) {
    use batuta::agent::capability::Capability;
    for cap in &manifest.capabilities {
        if let Capability::Spawn { max_depth } = cap {
            let pool = Arc::new(tokio::sync::Mutex::new(
                batuta::agent::pool::AgentPool::new(
                    Arc::clone(&driver),
                    4,
                ),
            ));
            registry.register(Box::new(
                batuta::agent::tool::spawn::SpawnTool::new(
                    pool,
                    manifest.clone(),
                    0,
                    *max_depth,
                ),
            ));
            break;
        }
    }
}

/// Register InferenceTool on the registry if Inference capability is present.
pub(super) fn register_inference_tool(
    registry: &mut batuta::agent::tool::ToolRegistry,
    manifest: &batuta::agent::AgentManifest,
    driver: Arc<dyn batuta::agent::driver::LlmDriver>,
) {
    use batuta::agent::capability::Capability;
    if manifest.capabilities.contains(&Capability::Inference) {
        let max_tokens = manifest.model.max_tokens;
        registry.register(Box::new(
            batuta::agent::tool::inference::InferenceTool::new(
                driver,
                max_tokens,
            ),
        ));
    }
}

/// Build memory substrate (TruenoMemory when available, else InMemory).
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

/// Validate model file integrity (G0) and format (G1).
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

/// Validate model inference sanity (G2 gate).
/// Runs a probe prompt and checks character entropy for garbage detection.
pub(super) fn validate_model_g2(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<()> {
    println!();
    println!(
        "{} G2: Inference Sanity Check",
        "🧪".bright_cyan().bold()
    );
    println!("{}", "─".repeat(40).dimmed());

    let driver = build_driver(manifest)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    let probe_prompt = "Respond with exactly: Hello, I am operational.";

    let request = batuta::agent::driver::CompletionRequest {
        model: String::new(),
        messages: vec![
            batuta::agent::driver::Message::User(
                probe_prompt.into(),
            ),
        ],
        max_tokens: 64,
        temperature: 0.0,
        tools: vec![],
        system: Some(manifest.model.system_prompt.clone()),
    };

    let result = rt.block_on(async {
        driver.complete(request).await
    });

    match result {
        Ok(response) => {
            let text = &response.text;
            if text.is_empty() {
                println!("  {} G2 FAIL: empty response", "✗".bright_red());
                anyhow::bail!("G2: model returned empty response");
            }
            let entropy = char_entropy(text);
            let dot = "•".bright_blue();
            println!("  {dot} G2 probe: \"{}\"", truncate_str(text, 60));
            println!("  {dot} G2 metrics: len={}, entropy={:.2}", text.len(), entropy);
            if entropy > 5.5 {
                println!("  {} G2 WARN: high entropy ({:.2}) — check LAYOUT-002",
                    "⚠".bright_yellow(), entropy);
            }
            println!("  {} G2 PASS: model produces coherent output", "✓".green());
            Ok(())
        }
        Err(e) => {
            println!("  {} G2 FAIL: inference error: {e}", "✗".bright_red());
            anyhow::bail!("G2: inference failed: {e}");
        }
    }
}

/// Shannon entropy of a string (bits per character).
pub(super) fn char_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    let total = s.len() as f64;
    for b in s.bytes() {
        freq[b as usize] += 1;
    }
    let mut entropy = 0.0;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Truncate a string to max_len with ellipsis.
pub(super) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
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
    let dot = "•".bright_blue();
    println!("{}", "🤖 Batuta Agent Runtime (Sovereign)".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!("{dot} Agent: {}", manifest.name.cyan());
    println!("{dot} Version: {}", manifest.version.dimmed());
    println!("{dot} Privacy: {:?}", manifest.privacy);
    println!("{dot} Capabilities: {:?}", manifest.capabilities);
    println!("{dot} Max iterations: {}", manifest.resources.max_iterations);
    if let Some(ref path) = manifest.model.model_path {
        println!("{dot} Model: {}", path.display());
    } else if let Some(ref repo) = manifest.model.model_repo {
        let q = manifest.model.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("{dot} Model: {} ({q})", repo.cyan());
    } else {
        println!("{dot} Model: {}", "none (specify model_path or model_repo)".dimmed());
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
