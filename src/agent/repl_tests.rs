//! Tests for the interactive REPL — slash commands, session tracking,
//! and conversation history compaction.
//! See: apr-code.md §3.3–3.4, PMAT-115.

use super::*;

#[test]
fn test_slash_command_parse() {
    assert_eq!(SlashCommand::parse("/help"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/h"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/?"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/quit"), Some(SlashCommand::Quit));
    assert_eq!(SlashCommand::parse("/q"), Some(SlashCommand::Quit));
    assert_eq!(SlashCommand::parse("/exit"), Some(SlashCommand::Quit));
    assert_eq!(SlashCommand::parse("/cost"), Some(SlashCommand::Cost));
    assert_eq!(SlashCommand::parse("/context"), Some(SlashCommand::Context));
    assert_eq!(SlashCommand::parse("/ctx"), Some(SlashCommand::Context));
    assert_eq!(SlashCommand::parse("/model"), Some(SlashCommand::Model));
    assert_eq!(SlashCommand::parse("/compact"), Some(SlashCommand::Compact));
    assert_eq!(SlashCommand::parse("/clear"), Some(SlashCommand::Clear));
    assert_eq!(SlashCommand::parse("/unknown"), Some(SlashCommand::Unknown("/unknown".into())));
}

#[test]
fn test_slash_command_parse_not_slash() {
    assert_eq!(SlashCommand::parse("hello"), None);
    assert_eq!(SlashCommand::parse(""), None);
    assert_eq!(SlashCommand::parse("help"), None);
}

#[test]
fn test_slash_command_parse_with_args() {
    assert_eq!(SlashCommand::parse("/help me"), Some(SlashCommand::Help));
    assert_eq!(SlashCommand::parse("/model gpt-4"), Some(SlashCommand::Model));
}

#[test]
fn test_repl_session_new() {
    let session = ReplSession::new("test");
    assert_eq!(session.turn_count, 0);
    assert_eq!(session.total_input_tokens, 0);
    assert_eq!(session.total_output_tokens, 0);
    assert_eq!(session.total_tool_calls, 0);
    assert_eq!(session.estimated_cost_usd, 0.0);
}

#[test]
fn test_repl_session_record_turn() {
    let mut session = ReplSession::new("test");
    let result = AgentLoopResult {
        text: "hello".into(),
        usage: crate::agent::result::TokenUsage { input_tokens: 100, output_tokens: 50 },
        iterations: 2,
        tool_calls: 3,
    };
    session.record_turn(&result, 0.005);

    assert_eq!(session.turn_count, 1);
    assert_eq!(session.total_input_tokens, 100);
    assert_eq!(session.total_output_tokens, 50);
    assert_eq!(session.total_tool_calls, 3);
    assert!((session.estimated_cost_usd - 0.005).abs() < 1e-10);

    // Second turn
    session.record_turn(&result, 0.003);
    assert_eq!(session.turn_count, 2);
    assert_eq!(session.total_input_tokens, 200);
    assert_eq!(session.total_output_tokens, 100);
    assert_eq!(session.total_tool_calls, 6);
}

#[test]
fn test_compact_history_short_noop() {
    let mut history = vec![Message::User("hello".into()), Message::Assistant("hi".into())];
    compact_history(&mut history);
    assert_eq!(history.len(), 2);
}

#[test]
fn test_compact_history_strips_old_tool_calls() {
    use crate::agent::driver::{ToolCall, ToolResultMsg};
    let mut history = Vec::new();
    for i in 0..4 {
        history.push(Message::User(format!("question {i}")));
        history.push(Message::AssistantToolUse(ToolCall {
            id: format!("c{i}"),
            name: "shell".into(),
            input: serde_json::json!({"command": "ls"}),
        }));
        history.push(Message::ToolResult(ToolResultMsg {
            tool_use_id: format!("c{i}"),
            content: "file1 file2".into(),
            is_error: false,
        }));
        history.push(Message::Assistant(format!("answer {i}")));
    }
    assert_eq!(history.len(), 16);

    compact_history(&mut history);

    assert!(history.len() < 16, "expected compaction, got {}", history.len());
    let users = history.iter().filter(|m| matches!(m, Message::User(_))).count();
    assert_eq!(users, 4, "all user messages preserved");
    let assts = history.iter().filter(|m| matches!(m, Message::Assistant(_))).count();
    assert_eq!(assts, 4, "all assistant messages preserved");
}

#[test]
fn test_compact_history_preserves_recent() {
    let mut history: Vec<Message> = (0..12)
        .map(|i| {
            if i % 2 == 0 {
                Message::User(format!("q{i}"))
            } else {
                Message::Assistant(format!("a{i}"))
            }
        })
        .collect();
    compact_history(&mut history);
    assert_eq!(history.len(), 12);
}

#[test]
fn test_slash_command_session_and_sessions() {
    assert_eq!(SlashCommand::parse("/session"), Some(SlashCommand::Session));
    assert_eq!(SlashCommand::parse("/sessions"), Some(SlashCommand::Sessions));
}

#[test]
fn test_session_persists_messages() {
    let session = ReplSession::new("test-persist");
    let history = vec![
        Message::User("hello".into()),
        Message::Assistant("hi there".into()),
        Message::User("bye".into()),
    ];

    // Persist messages (from index 0)
    session.persist_messages(&history, 0);

    // Load them back via the store
    if let Some(ref store) = session.store {
        let loaded = store.load_messages().expect("load");
        assert_eq!(loaded.len(), 3, "all messages persisted");
        // Cleanup
        let _ = std::fs::remove_dir_all(&store.dir);
    }
}

#[test]
fn test_session_incremental_persist() {
    let session = ReplSession::new("test-incremental");
    let mut history = vec![Message::User("turn1".into()), Message::Assistant("resp1".into())];

    // Persist first turn (0..2)
    session.persist_messages(&history, 0);

    // Add second turn
    history.push(Message::User("turn2".into()));
    history.push(Message::Assistant("resp2".into()));

    // Persist only new messages (2..4)
    session.persist_messages(&history, 2);

    if let Some(ref store) = session.store {
        let loaded = store.load_messages().expect("load");
        assert_eq!(loaded.len(), 4, "both turns persisted");
        let _ = std::fs::remove_dir_all(&store.dir);
    }
}
