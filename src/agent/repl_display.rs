//! Display helpers for the interactive REPL.
//!
//! Extracted from repl.rs to stay under the 500-line limit.
//! Contains welcome banner, help, turn footer, session summary,
//! and streaming event rendering.

use std::io::{self, Write};

use super::driver::{LlmDriver, StreamEvent};
use super::repl::ReplSession;
use super::result::AgentLoopResult;
use super::AgentManifest;
use crate::ansi_colors::Colorize;

/// Print a streaming event in REPL format.
pub(super) fn print_stream_event_repl(event: &StreamEvent) {
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
pub(super) fn print_welcome(manifest: &AgentManifest, driver: &dyn LlmDriver) {
    let tier = driver.privacy_tier();
    println!();
    println!(
        "  {} {} ({})",
        "apr code".bright_cyan().bold(),
        env!("CARGO_PKG_VERSION"),
        format!("{tier:?} tier").dimmed()
    );

    if let Some(ref path) = manifest.model.model_path {
        let name = path.file_name().map(|f| f.to_string_lossy()).unwrap_or_default();
        let fmt = match path.extension().and_then(|e| e.to_str()) {
            Some("apr") => "APR",
            Some("gguf") => "GGUF",
            _ => "model",
        };
        println!("  {} {} ({fmt})", "Model:".dimmed(), name.bright_cyan());
    } else {
        println!("  {} {}", "Model:".dimmed(), "mock (no model loaded)".bright_yellow());
    }
    println!();
    println!(
        "  Type a message, {} for commands, {} to exit.",
        "/help".bright_yellow(),
        "/quit".bright_yellow()
    );
    println!("  {}", "─".repeat(56).dimmed());
}

/// Print help for slash commands.
pub(super) fn print_help() {
    let cmds = [
        ("/help", "Show this help"),
        ("/test", "Run cargo test"),
        ("/quality", "Run quality gate"),
        ("/context", "Show context/token usage"),
        ("/compact", "Compact old messages"),
        ("/session", "Show session info"),
        ("/sessions", "List recent sessions"),
        ("/cost", "Show session cost"),
        ("/clear", "Clear screen + history"),
        ("/quit", "Exit apr code"),
    ];
    println!("  {}", "Commands:".bold());
    for (cmd, desc) in cmds {
        println!("    {:10} {desc}", cmd.bright_yellow());
    }
}

/// Print footer after each turn.
pub(super) fn print_turn_footer(
    result: &AgentLoopResult,
    cost: f64,
    session: &ReplSession,
    _budget: f64,
) {
    let cost_str = if cost > 0.0 { format!("${:.4}", cost) } else { "free".into() };
    println!(
        "\n{}",
        format!(
            "  [turn {} | {} | {} tools | {}/{} tok]",
            session.turn_count,
            cost_str,
            result.tool_calls,
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        .dimmed()
    );
}

/// Print session summary on exit.
pub(super) fn print_session_summary(session: &ReplSession) {
    if session.turn_count == 0 {
        return;
    }
    println!("\n  {}", "Session Summary".bold());
    println!("    Turns: {}, Tool calls: {}", session.turn_count, session.total_tool_calls);
    println!("    Tokens: {} in / {} out", session.total_input_tokens, session.total_output_tokens);
    if let Some(id) = session.session_id() {
        println!("    Session: {id}");
    }
}

/// List recent sessions for the current working directory.
pub(super) fn list_recent_sessions() {
    let sessions_dir = match dirs::home_dir() {
        Some(h) => h.join(".apr").join("sessions"),
        None => {
            println!("  Cannot determine home directory.");
            return;
        }
    };
    if !sessions_dir.is_dir() {
        println!("  No sessions found.");
        return;
    }

    let mut sessions: Vec<(String, u32, String)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let manifest_path = entry.path().join("manifest.json");
            if let Ok(json) = std::fs::read_to_string(&manifest_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                    let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("?").to_string();
                    let turns = v.get("turns").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
                    let cwd = v.get("cwd").and_then(|c| c.as_str()).unwrap_or("?").to_string();
                    sessions.push((id, turns, cwd));
                }
            }
        }
    }

    if sessions.is_empty() {
        println!("  No sessions found.");
        return;
    }
    sessions.sort_by(|a, b| b.0.cmp(&a.0));
    println!("  Recent sessions:");
    for (id, turns, cwd) in sessions.iter().take(10) {
        println!("    {} ({turns} turns) {}", id.cyan(), cwd.dimmed());
    }
    println!("  Resume: {} --resume=<id>", "batuta code".bright_yellow());
}

/// Run a shell command as a slash command shortcut.
pub(super) fn run_shell_shortcut(cmd: &str) {
    match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stdout.is_empty() {
                println!("{stdout}");
            }
            if !stderr.is_empty() {
                eprintln!("{stderr}");
            }
            if output.status.success() {
                println!("  {} Done.", "✓".green());
            } else {
                println!("  {} Exit code: {}", "✗".bright_red(), output.status);
            }
        }
        Err(e) => println!("  {} Failed: {e}", "✗".bright_red()),
    }
}

/// Compact conversation history by removing tool call/result details
/// from older turns, keeping only the user queries and assistant summaries.
pub(super) fn compact_history(history: &mut Vec<super::driver::Message>) {
    use super::driver::Message;
    if history.len() <= 10 {
        return;
    }
    let compact_boundary = history.len() - 10;
    let mut compacted = Vec::new();
    for msg in history.iter().take(compact_boundary) {
        match msg {
            Message::User(_) | Message::Assistant(_) => compacted.push(msg.clone()),
            Message::AssistantToolUse(_) | Message::ToolResult(_) => {}
            Message::System(_) => compacted.push(msg.clone()),
        }
    }
    compacted.extend_from_slice(&history[compact_boundary..]);
    *history = compacted;
}
