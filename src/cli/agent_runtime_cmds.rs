//! Agent runtime command handlers (run, chat, pool).
//!
//! Extracted from `cli/agent.rs` for QA-002 compliance (≤500 lines).

use std::path::PathBuf;
use std::sync::Arc;

use crate::ansi_colors::Colorize;

use super::agent_helpers::{
    build_driver, build_guard, build_memory, build_tool_registry,
    load_manifest, print_manifest_summary, print_stream_event,
    register_inference_tool, register_spawn_tool, try_auto_pull,
};
#[cfg(feature = "agents-mcp")]
use super::agent_helpers::register_mcp_tools;

/// Run an agent with a single prompt.
pub(super) fn cmd_agent_run(
    manifest_path: &PathBuf,
    prompt: &str,
    max_iterations: Option<u32>,
    daemon: bool,
    auto_pull: bool,
    stream: bool,
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
    let driver: Arc<dyn batuta::agent::driver::LlmDriver> =
        Arc::from(build_driver(&manifest)?);

    // Register tools based on manifest capabilities
    let mut tools = build_tool_registry(&manifest);
    register_spawn_tool(&mut tools, &manifest, Arc::clone(&driver));
    register_inference_tool(&mut tools, &manifest, Arc::clone(&driver));
    #[cfg(feature = "agents-mcp")]
    rt.block_on(register_mcp_tools(&mut tools, &manifest));

    // Memory substrate
    let memory = build_memory();

    // Stream events to stdout
    let (tx, rx) =
        tokio::sync::mpsc::channel::<batuta::agent::driver::StreamEvent>(64);

    println!("{} Starting agent loop...", "▶".bright_green());
    println!("{}\n", "─".repeat(60).dimmed());

    let result = rt.block_on(run_loop_async(
        &manifest, prompt, driver.as_ref(), &tools, memory.as_ref(), tx, rx, stream,
    ));

    println!("\n{}", "─".repeat(60).dimmed());

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
pub(super) fn cmd_agent_chat(
    manifest_path: &PathBuf,
    auto_pull: bool,
    stream: bool,
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
        let input = match read_chat_input(&stdin, &mut line_buf) {
            ChatInput::Line(s) => s,
            ChatInput::Exit => break,
            ChatInput::Empty => continue,
        };

        let result = run_chat_turn(
            &rt, &manifest, &input, driver.as_ref(),
            &tools, memory.as_ref(), stream,
        );
        print_chat_result(&result);
    }

    Ok(())
}

/// Outcome of reading one line of chat input.
enum ChatInput {
    Line(String),
    Exit,
    Empty,
}

/// Read one line of chat input from stdin.
fn read_chat_input(
    stdin: &std::io::Stdin,
    buf: &mut String,
) -> ChatInput {
    print!("\n{} ", "You>".bright_green().bold());
    use std::io::Write;
    std::io::stdout().flush().ok();

    buf.clear();
    let bytes = match stdin.read_line(buf) {
        Ok(b) => b,
        Err(_) => return ChatInput::Exit,
    };
    if bytes == 0 {
        println!("\n{} Goodbye.", "✓".green());
        return ChatInput::Exit;
    }
    let input = buf.trim();
    if input.is_empty() {
        return ChatInput::Empty;
    }
    if input == "quit" || input == "exit" {
        println!("{} Goodbye.", "✓".green());
        return ChatInput::Exit;
    }
    ChatInput::Line(input.to_string())
}

/// Execute one chat turn (agent loop invocation).
fn run_chat_turn(
    rt: &tokio::runtime::Runtime,
    manifest: &batuta::agent::AgentManifest,
    input: &str,
    driver: &dyn batuta::agent::driver::LlmDriver,
    tools: &batuta::agent::tool::ToolRegistry,
    memory: &dyn batuta::agent::memory::MemorySubstrate,
    stream: bool,
) -> Result<batuta::agent::AgentLoopResult, batuta::agent::AgentError> {
    if stream {
        let (tx, rx) =
            tokio::sync::mpsc::channel::<batuta::agent::driver::StreamEvent>(64);
        rt.block_on(run_loop_async(
            manifest, input, driver, tools, memory, tx, rx, true,
        ))
    } else {
        rt.block_on(batuta::agent::runtime::run_agent_loop(
            manifest, input, driver, tools, memory, None,
        ))
    }
}

/// Print the result of a chat turn.
fn print_chat_result(
    result: &Result<batuta::agent::AgentLoopResult, batuta::agent::AgentError>,
) {
    match result {
        Ok(r) => {
            println!(
                "\n{} {}",
                "Agent>".bright_cyan().bold(),
                r.text
            );
            println!(
                "{}",
                format!(
                    "  [iter={}, tools={}, tokens={}/{}]",
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
                "\n{} Error: {e}",
                "✗".bright_red()
            );
        }
    }
}

/// Fan-out multiple agents and collect results.
pub(super) fn cmd_agent_pool(
    manifest_paths: &[PathBuf],
    prompt: &str,
    concurrency: Option<usize>,
) -> anyhow::Result<()> {
    use batuta::agent::pool::{AgentPool, SpawnConfig};

    let manifests: Vec<batuta::agent::AgentManifest> = manifest_paths
        .iter()
        .map(load_manifest)
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

/// Execute agent loop with optional streaming.
async fn run_loop_async(
    manifest: &batuta::agent::AgentManifest,
    prompt: &str,
    driver: &dyn batuta::agent::driver::LlmDriver,
    tools: &batuta::agent::tool::ToolRegistry,
    memory: &dyn batuta::agent::memory::MemorySubstrate,
    tx: tokio::sync::mpsc::Sender<batuta::agent::driver::StreamEvent>,
    mut rx: tokio::sync::mpsc::Receiver<batuta::agent::driver::StreamEvent>,
    stream: bool,
) -> Result<batuta::agent::AgentLoopResult, batuta::agent::AgentError> {
    if stream {
        let drain = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                print_stream_event(&event);
            }
        });
        let r = batuta::agent::runtime::run_agent_loop(
            manifest, prompt, driver, tools, memory, Some(tx),
        ).await;
        let _ = drain.await;
        r
    } else {
        let r = batuta::agent::runtime::run_agent_loop(
            manifest, prompt, driver, tools, memory, Some(tx),
        ).await;
        while let Ok(event) = rx.try_recv() {
            print_stream_event(&event);
        }
        r
    }
}

/// Display agent manifest status and configuration.
pub(super) fn cmd_agent_status(manifest_path: &PathBuf) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;
    if let Err(errors) = manifest.validate() {
        println!("{} Manifest has validation errors:", "⚠".bright_yellow());
        for e in &errors {
            println!("  {} {e}", "✗".red());
        }
        println!();
    }
    print_manifest_summary(&manifest);
    println!();

    let r = &manifest.resources;
    println!("{}", "Resource Quotas".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!("  Max iterations:  {}", r.max_iterations);
    println!("  Max tool calls:  {}", r.max_tool_calls);
    let budget = if r.max_cost_usd > 0.0 {
        format!("${:.4}", r.max_cost_usd)
    } else {
        "unlimited (sovereign)".to_string()
    };
    println!("  Cost budget:     {budget}");
    println!();

    let m = &manifest.model;
    println!("{}", "Model Configuration".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!("  Max tokens:      {}", m.max_tokens);
    println!("  Temperature:     {}", m.temperature);
    match m.context_window {
        Some(ref ctx) => println!("  Context window:  {ctx}"),
        None => println!("  Context window:  auto-detect"),
    }
    if let Some(ref repo) = m.model_repo {
        let quant = m.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("  Model repo:      {repo}");
        println!("  Quantization:    {quant}");
        if let Some(resolved) = m.resolve_model_path() {
            let exists = if resolved.exists() { "found" } else { "not found" };
            println!("  Resolved path:   {} ({exists})", resolved.display());
        }
    }
    if let Some(ref remote) = m.remote_model {
        println!("  Remote model:    {remote}");
    }
    println!();

    println!("{}", "Capabilities".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    if manifest.capabilities.is_empty() {
        println!("  (none — deny-by-default)");
    }
    for cap in &manifest.capabilities {
        println!("  {} {cap:?}", "•".bright_blue());
    }
    println!();
    println!("{} {}", "✓".green(), "Agent manifest loaded successfully.".green());
    Ok(())
}
