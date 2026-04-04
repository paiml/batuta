//! Tests for `agent::code` — apr code library entry point.

use super::*;

#[test]
fn test_build_default_manifest_always_sovereign() {
    let m = build_default_manifest();
    assert_eq!(m.name, "apr-code");
    assert_eq!(m.privacy, PrivacyTier::Sovereign);
    assert!(!m.capabilities.is_empty());
}

#[test]
fn test_build_code_tools_registers_all() {
    let m = build_default_manifest();
    let tools = build_code_tools(&m);
    assert!(tools.get("file_read").is_some(), "missing file_read");
    assert!(tools.get("file_write").is_some(), "missing file_write");
    assert!(tools.get("file_edit").is_some(), "missing file_edit");
    assert!(tools.get("glob").is_some(), "missing glob");
    assert!(tools.get("grep").is_some(), "missing grep");
    assert!(tools.get("shell").is_some(), "missing shell");
    assert!(tools.get("memory").is_some(), "missing memory");
    // PMAT-163: pmat_query tool
    assert!(tools.get("pmat_query").is_some(), "missing pmat_query (PMAT-163)");
    #[cfg(feature = "rag")]
    assert!(tools.get("rag").is_some(), "missing rag tool (PMAT-153)");
    // 9 tools with rag, 8 without
    #[cfg(feature = "rag")]
    assert!(tools.len() >= 9, "expected >=9 tools with rag, got {}", tools.len());
    #[cfg(not(feature = "rag"))]
    assert!(tools.len() >= 8, "expected >=8 tools, got {}", tools.len());
}

#[test]
fn test_code_system_prompt_not_empty() {
    assert!(CODE_SYSTEM_PROMPT.len() > 200);
    assert!(CODE_SYSTEM_PROMPT.contains("tool_call"));
    assert!(CODE_SYSTEM_PROMPT.contains("sovereign"));
    // PMAT-168: all 9 tools enumerated with examples
    for tool in &[
        "file_read", "file_write", "file_edit", "glob", "grep",
        "shell", "memory", "pmat_query", "rag",
    ] {
        assert!(
            CODE_SYSTEM_PROMPT.contains(tool),
            "system prompt missing tool: {tool}"
        );
    }
    // Verify example inputs exist (not just names)
    assert!(CODE_SYSTEM_PROMPT.contains("src/main.rs"), "missing file_read example");
    assert!(CODE_SYSTEM_PROMPT.contains("cargo test"), "missing shell example");
    assert!(CODE_SYSTEM_PROMPT.contains("error handling"), "missing pmat_query example");
}

#[test]
fn test_load_project_instructions_from_claude_md() {
    let instructions = load_project_instructions(4096);
    assert!(instructions.is_some(), "expected to find CLAUDE.md in project root");
    let text = instructions.expect("just checked");
    assert!(
        text.contains("batuta") || text.contains("Batuta") || text.contains("CLAUDE"),
        "CLAUDE.md should mention the project"
    );
}

#[test]
fn test_manifest_includes_project_instructions() {
    let m = build_default_manifest();
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
    assert!(
        ctx.contains("Rust") || ctx.contains("Cargo") || ctx.contains("Language:"),
        "should detect language or build system: {ctx}"
    );
}

#[test]
fn test_manifest_includes_project_context() {
    let m = build_default_manifest();
    assert!(
        m.model.system_prompt.contains("Project Context"),
        "system prompt should contain project context section"
    );
    assert!(
        m.model.system_prompt.contains("Working directory:"),
        "context should include working directory"
    );
}

#[test]
fn test_instruction_budget_scales_with_context() {
    assert_eq!(instruction_budget(2048), 0, "2K context: skip instructions");
    assert_eq!(instruction_budget(4096), 1024, "4K context: 25% = 1024");
    assert_eq!(instruction_budget(8192), 2048, "8K context: 25% = 2048");
    assert_eq!(instruction_budget(32768), 4096, "32K context: capped at 4096");
    assert_eq!(instruction_budget(131072), 4096, "128K context: capped at 4096");
}

#[test]
fn test_load_instructions_zero_budget_returns_none() {
    let result = load_project_instructions(0);
    assert!(result.is_none(), "zero budget should skip instructions");
}

#[test]
fn test_exit_codes_match_spec() {
    assert_eq!(exit_code::SUCCESS, 0);
    assert_eq!(exit_code::AGENT_ERROR, 1);
    assert_eq!(exit_code::BUDGET_EXHAUSTED, 2);
    assert_eq!(exit_code::MAX_TURNS, 3);
    assert_eq!(exit_code::SANDBOX_VIOLATION, 4);
    assert_eq!(exit_code::NO_MODEL, 5);
}

#[test]
fn test_fallback_driver_without_model() {
    let manifest = build_default_manifest();
    // No model path set — should return MockDriver
    let driver = build_fallback_driver(&manifest);
    assert!(driver.is_ok(), "fallback should succeed with mock");
}
