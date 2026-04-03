//! Interactive REPL for `apr code`.
//!
//! Provides a terminal UI where the user types prompts and the agent
//! streams responses token-by-token. Uses crossterm for raw mode
//! input and presentar-terminal patterns for output.
//!
//! Architecture:
//! - Main thread: crossterm event loop (keyboard input)
//! - Tokio task: agent loop (LLM + tools)
//! - mpsc channel: StreamEvent from agent → REPL display
//! - Arc<AtomicBool>: Ctrl+C cancels current generation
//!
//! See: apr-code.md §3, agent-and-playbook.md §7

use std::io::{self, Write};
use std::sync::Arc;

use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use crate::agent::driver::{LlmDriver, Message, StreamEvent};
use crate::agent::memory::MemorySubstrate;
use crate::agent::result::AgentLoopResult;
use crate::agent::tool::ToolRegistry;
use crate::agent::AgentManifest;
use crate::ansi_colors::Colorize;

/// Slash commands recognized by the REPL.
#[derive(Debug, PartialEq)]
enum SlashCommand {
    Help,
    Quit,
    Cost,
    Context,
    Model,
    Compact,
    Clear,
    Unknown(String),
}

impl SlashCommand {
    fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let cmd = trimmed.split_whitespace().next().unwrap_or("");
        Some(match cmd {
            "/help" | "/h" | "/?" => Self::Help,
            "/quit" | "/q" | "/exit" => Self::Quit,
            "/cost" => Self::Cost,
            "/context" | "/ctx" => Self::Context,
            "/model" => Self::Model,
            "/compact" => Self::Compact,
            "/clear" => Self::Clear,
            other => Self::Unknown(other.to_string()),
        })
    }
}

/// Session state tracked across turns.
struct ReplSession {
    turn_count: u32,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_tool_calls: u32,
    estimated_cost_usd: f64,
}

impl ReplSession {
    fn new() -> Self {
        Self {
            turn_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tool_calls: 0,
            estimated_cost_usd: 0.0,
        }
    }

    fn record_turn(&mut self, result: &AgentLoopResult, cost: f64) {
        self.turn_count += 1;
        self.total_input_tokens += result.usage.input_tokens;
        self.total_output_tokens += result.usage.output_tokens;
        self.total_tool_calls += result.tool_calls;
        self.estimated_cost_usd += cost;
    }
}

/// Run the interactive REPL.
///
/// This is the main entry point for `apr code` interactive mode.
/// Returns when the user types `/quit` or Ctrl+D.
pub fn run_repl(
    manifest: &AgentManifest,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    max_turns: u32,
    budget_usd: f64,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    print_welcome(manifest, driver);

    let mut session = ReplSession::new();
    let mut history: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut line_buf = String::new();

    loop {
        // Check turn budget
        if session.turn_count >= max_turns {
            println!(
                "\n{} Max turns ({}) reached. Session complete.",
                "⚠".bright_yellow(),
                max_turns
            );
            break;
        }
        if session.estimated_cost_usd >= budget_usd {
            println!(
                "\n{} Budget (${:.2}) exhausted. Session complete.",
                "⚠".bright_yellow(),
                budget_usd
            );
            break;
        }

        // Read input
        let input = match read_input(&stdin, &mut line_buf, &session, budget_usd, &mut history) {
            InputResult::Prompt(s) => s,
            InputResult::SlashHandled => continue,
            InputResult::Exit => break,
            InputResult::Empty => continue,
        };

        // Execute turn with streaming
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);

        // Install Ctrl+C handler for this turn
        rt.block_on(async {
            let flag = cancel_clone;
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    flag.store(true, Ordering::SeqCst);
                }
            });
        });

        let (tx, rx) = mpsc::channel::<StreamEvent>(64);

        println!();

        let result = rt
            .block_on(run_turn_streaming(manifest, &input, driver, tools, memory, &mut history, tx, rx, &cancel));

        match result {
            Ok(r) => {
                let cost = driver.estimate_cost(&r.usage);
                session.record_turn(&r, cost);
                print_turn_footer(&r, cost, &session, budget_usd);
            }
            Err(e) => {
                if cancel.load(Ordering::SeqCst) {
                    println!("\n{} Generation cancelled.", "⚠".bright_yellow());
                } else {
                    println!("\n{} Error: {e}", "✗".bright_red());
                }
            }
        }
    }

    print_session_summary(&session);
    Ok(())
}

/// Input reading result.
enum InputResult {
    Prompt(String),
    SlashHandled,
    Exit,
    Empty,
}

/// Read one line of input, handling slash commands inline.
fn read_input(
    stdin: &io::Stdin,
    buf: &mut String,
    session: &ReplSession,
    budget: f64,
    history: &mut Vec<Message>,
) -> InputResult {
    let cost_str = if session.estimated_cost_usd > 0.0 {
        format!(" ${:.3}", session.estimated_cost_usd)
    } else {
        String::new()
    };
    print!(
        "\n{}{} ",
        format!("[{}/{}{}]", session.turn_count + 1, "?", cost_str).dimmed(),
        " >".bright_green().bold(),
    );
    io::stdout().flush().ok();

    buf.clear();
    let bytes = match stdin.read_line(buf) {
        Ok(b) => b,
        Err(_) => return InputResult::Exit,
    };
    if bytes == 0 {
        println!();
        return InputResult::Exit;
    }

    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return InputResult::Empty;
    }

    // Handle slash commands
    if let Some(cmd) = SlashCommand::parse(trimmed) {
        handle_slash_command(&cmd, session, budget, history);
        return match cmd {
            SlashCommand::Quit => InputResult::Exit,
            _ => InputResult::SlashHandled,
        };
    }

    InputResult::Prompt(trimmed.to_string())
}

/// Handle a slash command.
fn handle_slash_command(cmd: &SlashCommand, session: &ReplSession, budget: f64, history: &mut Vec<Message>) {
    match cmd {
        SlashCommand::Help => print_help(),
        SlashCommand::Quit => println!("{} Goodbye.", "✓".green()),
        SlashCommand::Cost => {
            println!(
                "  Session cost: ${:.4} / ${:.2} ({:.1}%)",
                session.estimated_cost_usd,
                budget,
                (session.estimated_cost_usd / budget * 100.0).min(100.0)
            );
            println!(
                "  Tokens: {} in / {} out",
                session.total_input_tokens, session.total_output_tokens
            );
            println!("  Turns: {}, Tool calls: {}", session.turn_count, session.total_tool_calls);
        }
        SlashCommand::Context => {
            let user_msgs = history.iter().filter(|m| matches!(m, Message::User(_))).count();
            let asst_msgs = history.iter().filter(|m| matches!(m, Message::Assistant(_))).count();
            let tool_msgs = history.iter().filter(|m| matches!(m, Message::AssistantToolUse(_) | Message::ToolResult(_))).count();
            println!("  History: {} messages ({} user, {} assistant, {} tool)",
                history.len(), user_msgs, asst_msgs, tool_msgs);
            println!("  Turns: {}", session.turn_count);
        }
        SlashCommand::Model => {
            println!("  Model switching not yet implemented.");
        }
        SlashCommand::Compact => {
            let before = history.len();
            compact_history(history);
            println!("  Compacted: {} -> {} messages", before, history.len());
        }
        SlashCommand::Clear => {
            history.clear();
            print!("\x1B[2J\x1B[1;1H");
            io::stdout().flush().ok();
            println!("  Screen and conversation history cleared.");
        }
        SlashCommand::Unknown(name) => {
            println!("  {} Unknown command: {name}. Type /help for commands.", "?".bright_yellow());
        }
    }
}

/// Compact conversation history by removing tool call/result details
/// from older turns, keeping only the user queries and assistant summaries.
fn compact_history(history: &mut Vec<Message>) {
    if history.len() <= 10 {
        return; // Nothing to compact
    }
    // Keep last 10 messages intact, compact earlier ones
    let compact_boundary = history.len() - 10;
    let mut compacted = Vec::new();
    for msg in history.iter().take(compact_boundary) {
        match msg {
            Message::User(_) | Message::Assistant(_) => compacted.push(msg.clone()),
            // Drop tool details from old turns to save context
            Message::AssistantToolUse(_) | Message::ToolResult(_) => {}
            Message::System(_) => compacted.push(msg.clone()),
        }
    }
    // Append recent messages unchanged
    compacted.extend_from_slice(&history[compact_boundary..]);
    *history = compacted;
}

/// Execute one turn with streaming output and multi-turn history.
async fn run_turn_streaming(
    manifest: &AgentManifest,
    prompt: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    history: &mut Vec<Message>,
    tx: mpsc::Sender<StreamEvent>,
    mut rx: mpsc::Receiver<StreamEvent>,
    cancel: &Arc<AtomicBool>,
) -> Result<AgentLoopResult, crate::agent::result::AgentError> {
    // Drain task: print streaming events as they arrive
    let drain = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            print_stream_event_repl(&event);
        }
    });

    let result =
        crate::agent::runtime::run_agent_turn(manifest, history, prompt, driver, tools, memory, Some(tx))
            .await;

    // If cancelled, wrap the error
    if cancel.load(Ordering::SeqCst) && result.is_err() {
        return Err(crate::agent::result::AgentError::CircuitBreak("cancelled by user".into()));
    }

    // Ensure drain task finishes
    let _ = drain.await;
    result
}

/// Print a streaming event in REPL format.
fn print_stream_event_repl(event: &StreamEvent) {
    use crate::agent::phase::LoopPhase;
    match event {
        StreamEvent::PhaseChange { phase } => match phase {
            LoopPhase::Perceive => print!("{} ", "  [perceive]".dimmed()),
            LoopPhase::Reason => {}
            LoopPhase::Act { tool_name } => {
                println!("  {} {}", "  [tool]".bright_yellow(), tool_name.cyan());
            }
            LoopPhase::Done => {}
            LoopPhase::Error { message } => {
                println!("  {} {}", "  [error]".bright_red(), message);
            }
        },
        StreamEvent::TextDelta { text } => {
            print!("{text}");
            io::stdout().flush().ok();
        }
        StreamEvent::ToolUseStart { name, .. } => {
            print!("  {} {} ", "⚙".bright_yellow(), name.cyan());
            io::stdout().flush().ok();
        }
        StreamEvent::ToolUseEnd { result, .. } => {
            let preview =
                if result.len() > 60 { format!("{}...", &result[..57]) } else { result.clone() };
            println!("{}", preview.dimmed());
        }
        StreamEvent::ContentComplete { .. } => println!(),
    }
}

/// Print welcome banner.
fn print_welcome(manifest: &AgentManifest, driver: &dyn LlmDriver) {
    let tier = driver.privacy_tier();
    println!();
    println!(
        "  {} {} ({})",
        "apr code".bright_cyan().bold(),
        env!("CARGO_PKG_VERSION"),
        format!("{tier:?} tier").dimmed()
    );

    // Show model info
    if let Some(ref path) = manifest.model.model_path {
        let name = path.file_name().map(|f| f.to_string_lossy()).unwrap_or_default();
        let ext = path.extension().map(|e| e.to_string_lossy()).unwrap_or_default();
        let format_tag = if ext == "apr" { "APR" } else if ext == "gguf" { "GGUF" } else { "model" };
        println!("  {} {} ({})", "Model:".dimmed(), name.bright_cyan(), format_tag);
    } else {
        println!("  {} {}", "Model:".dimmed(), "mock (no model loaded)".bright_yellow());
    }

    println!("  {} {}", "Agent:".dimmed(), manifest.name);
    println!();
    println!(
        "  Type a message, {} for commands, {} to exit.",
        "/help".bright_yellow(),
        "/quit".bright_yellow()
    );
    println!("  {}", "─".repeat(56).dimmed());
}

/// Print help for slash commands.
fn print_help() {
    println!("  {}", "Commands:".bold());
    println!("    {}   Show this help", "/help".bright_yellow());
    println!("    {}   Show session cost", "/cost".bright_yellow());
    println!("    {}  Show context usage", "/context".bright_yellow());
    println!("    {}   Clear screen", "/clear".bright_yellow());
    println!("    {}   Exit apr code", "/quit".bright_yellow());
}

/// Print footer after each turn.
fn print_turn_footer(result: &AgentLoopResult, cost: f64, session: &ReplSession, budget: f64) {
    let cost_str = if cost > 0.0 { format!("${:.4}", cost) } else { "free".into() };
    let budget_pct = (session.estimated_cost_usd / budget * 100.0).min(100.0);
    println!(
        "\n{}",
        format!(
            "  [turn {} | {} | {} tools | {}/{} tok | {:.0}% budget]",
            session.turn_count,
            cost_str,
            result.tool_calls,
            result.usage.input_tokens,
            result.usage.output_tokens,
            budget_pct,
        )
        .dimmed()
    );
}

/// Print session summary on exit.
fn print_session_summary(session: &ReplSession) {
    if session.turn_count == 0 {
        return;
    }
    println!();
    println!("  {}", "Session Summary".bold());
    println!("    Turns: {}", session.turn_count);
    println!("    Tool calls: {}", session.total_tool_calls);
    println!("    Tokens: {} in / {} out", session.total_input_tokens, session.total_output_tokens);
    if session.estimated_cost_usd > 0.0 {
        println!("    Cost: ${:.4}", session.estimated_cost_usd);
    }
}

#[cfg(test)]
#[path = "repl_tests.rs"]
mod tests;
