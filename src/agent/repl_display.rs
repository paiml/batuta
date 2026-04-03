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
        ("/cost", "Show session cost"),
        ("/context", "Show context usage"),
        ("/compact", "Compact old messages"),
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
