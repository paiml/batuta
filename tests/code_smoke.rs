//! Smoke test: `batuta code` end-to-end pipeline with MockDriver.
//!
//! Proves the full agent loop works: MockDriver → tool_use(file_read) → tool
//! executes → final response. This is the minimum viability test for apr code.
//!
//! Refs: PMAT-107, apr-code-feasibility-falsification.md §7 F-1

#![cfg(feature = "agents")]

use std::path::PathBuf;

use batuta::agent::capability::Capability;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use batuta::agent::tool::search::{GlobTool, GrepTool};
use batuta::agent::tool::shell::ShellTool;
use batuta::agent::tool::ToolRegistry;
use batuta::serve::backends::PrivacyTier;

/// Build the same default manifest as `cli::code::build_default_manifest`.
fn code_manifest() -> AgentManifest {
    AgentManifest {
        name: "apr-code-test".to_string(),
        description: "Smoke test".to_string(),
        privacy: PrivacyTier::Standard,
        model: ModelConfig {
            system_prompt: "You are a coding assistant.".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            ..ModelConfig::default()
        },
        resources: ResourceQuota {
            max_iterations: 10,
            max_tool_calls: 20,
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

/// Build the same tool registry as `cli::code::build_code_tools`.
fn code_tools() -> ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(FileReadTool::new(vec!["*".into()])));
    tools.register(Box::new(FileWriteTool::new(vec!["*".into()])));
    tools.register(Box::new(FileEditTool::new(vec!["*".into()])));
    tools.register(Box::new(GlobTool::new(vec!["*".into()])));
    tools.register(Box::new(GrepTool::new(vec!["*".into()])));
    tools.register(Box::new(ShellTool::new(vec!["*".into()], cwd)));
    tools
}

/// Smoke test: agent loop with MockDriver that reads a file.
#[tokio::test]
async fn test_code_smoke_file_read() {
    // Create a temp file to read
    let dir = tempfile::TempDir::new().unwrap();
    let test_file = dir.path().join("hello.rs");
    std::fs::write(&test_file, "fn main() { println!(\"hello\"); }\n").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    // MockDriver: first response is tool_use(file_read), second is final text
    let driver = MockDriver::tool_then_response(
        "file_read",
        serde_json::json!({"path": test_file.to_str().unwrap()}),
        "The file contains a hello world program.",
    );

    let result = run_agent_loop(&manifest, "Read hello.rs", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "The file contains a hello world program.");
    assert!(result.iterations >= 2, "expected at least 2 iterations (tool_use + end_turn)");
    assert_eq!(result.tool_calls, 1, "expected 1 tool call");
}

/// Smoke test: agent loop with MockDriver that writes a file.
#[tokio::test]
async fn test_code_smoke_file_write() {
    let dir = tempfile::TempDir::new().unwrap();
    let target_file = dir.path().join("output.txt");

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "file_write",
        serde_json::json!({
            "path": target_file.to_str().unwrap(),
            "content": "Hello from apr code!"
        }),
        "File written successfully.",
    );

    let result = run_agent_loop(&manifest, "Write output.txt", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "File written successfully.");
    assert!(target_file.exists(), "file was not created");
    assert_eq!(std::fs::read_to_string(&target_file).unwrap(), "Hello from apr code!");
}

/// Smoke test: agent loop with MockDriver that edits a file.
#[tokio::test]
async fn test_code_smoke_file_edit() {
    let dir = tempfile::TempDir::new().unwrap();
    let test_file = dir.path().join("code.rs");
    std::fs::write(&test_file, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "file_edit",
        serde_json::json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "println!(\"hello\")",
            "new_string": "println!(\"world\")"
        }),
        "Changed hello to world.",
    );

    let result = run_agent_loop(&manifest, "Fix the greeting", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "Changed hello to world.");
    let content = std::fs::read_to_string(&test_file).unwrap();
    assert!(content.contains("println!(\"world\")"), "edit not applied");
    assert!(!content.contains("println!(\"hello\")"), "old string still present");
}

/// Smoke test: agent loop with MockDriver that runs shell command.
#[tokio::test]
async fn test_code_smoke_shell() {
    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "shell",
        serde_json::json!({"command": "echo hello_from_agent"}),
        "Shell returned hello.",
    );

    let result =
        run_agent_loop(&manifest, "Run echo", &driver, &tools, &memory, None).await.expect("fail");

    assert_eq!(result.text, "Shell returned hello.");
    assert_eq!(result.tool_calls, 1);
}

/// Smoke test: agent loop with MockDriver that uses glob.
#[tokio::test]
async fn test_code_smoke_glob() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.rs"), "").unwrap();
    std::fs::write(dir.path().join("b.rs"), "").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "glob",
        serde_json::json!({
            "pattern": "*.rs",
            "path": dir.path().to_str().unwrap()
        }),
        "Found 2 rust files.",
    );

    let result = run_agent_loop(&manifest, "Find rs files", &driver, &tools, &memory, None)
        .await
        .expect("fail");

    assert_eq!(result.text, "Found 2 rust files.");
}

/// Smoke test: verify tool count matches spec (7 tools).
#[test]
fn test_code_tool_count() {
    let tools = code_tools();
    // file_read, file_write, file_edit, glob, grep, shell = 6
    // (memory is registered separately in cli::code, not in code_tools() here)
    assert_eq!(tools.len(), 6, "expected 6 tools in base code_tools()");
}

/// Smoke test: simple end_turn without tool use.
#[tokio::test]
async fn test_code_smoke_no_tools() {
    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();
    let driver = MockDriver::single_response("I can help you with coding tasks.");

    let result =
        run_agent_loop(&manifest, "Hello", &driver, &tools, &memory, None).await.expect("fail");

    assert_eq!(result.text, "I can help you with coding tasks.");
    assert_eq!(result.tool_calls, 0);
    assert_eq!(result.iterations, 1);
}
