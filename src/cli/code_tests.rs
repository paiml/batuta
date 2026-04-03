//! Tests for `batuta code` CLI module.

use super::*;

#[test]
fn test_build_default_manifest_always_sovereign() {
    let m1 = build_default_manifest(false);
    assert_eq!(m1.name, "apr-code");
    assert_eq!(m1.privacy, PrivacyTier::Sovereign);
    assert!(!m1.capabilities.is_empty());

    let m2 = build_default_manifest(true);
    assert_eq!(m2.privacy, PrivacyTier::Sovereign);
}

#[test]
fn test_build_code_tools_registers_all() {
    let m = build_default_manifest(false);
    let tools = build_code_tools(&m);
    assert!(tools.get("file_read").is_some(), "missing file_read");
    assert!(tools.get("file_write").is_some(), "missing file_write");
    assert!(tools.get("file_edit").is_some(), "missing file_edit");
    assert!(tools.get("glob").is_some(), "missing glob");
    assert!(tools.get("grep").is_some(), "missing grep");
    assert!(tools.get("shell").is_some(), "missing shell");
    assert!(tools.get("memory").is_some(), "missing memory");
    // PMAT-153: RAG tool wired (8 tools with rag feature)
    #[cfg(feature = "rag")]
    assert!(tools.get("rag").is_some(), "missing rag tool (PMAT-153)");
    #[cfg(feature = "rag")]
    assert!(tools.len() >= 8, "expected >=8 tools with rag, got {}", tools.len());
    #[cfg(not(feature = "rag"))]
    assert!(tools.len() >= 7, "expected >=7 tools, got {}", tools.len());
}

#[test]
fn test_default_manifest_is_sovereign() {
    let m = build_default_manifest(true);
    assert_eq!(m.privacy, PrivacyTier::Sovereign);
    let m2 = build_default_manifest(false);
    assert_eq!(m2.privacy, PrivacyTier::Sovereign, "apr code must always be Sovereign");
}

#[test]
fn test_code_system_prompt_not_empty() {
    assert!(CODE_SYSTEM_PROMPT.len() > 200);
    assert!(CODE_SYSTEM_PROMPT.contains("tool_call"));
    assert!(CODE_SYSTEM_PROMPT.contains("file_read"));
    assert!(CODE_SYSTEM_PROMPT.contains("file_edit"));
    assert!(CODE_SYSTEM_PROMPT.contains("shell"));
    assert!(CODE_SYSTEM_PROMPT.contains("APR"));
    assert!(CODE_SYSTEM_PROMPT.contains("sovereign"));
}

#[test]
fn test_load_project_instructions_from_claude_md() {
    let instructions = load_project_instructions(4096);
    assert!(instructions.is_some(), "expected to find CLAUDE.md in project root");
    let text = instructions.unwrap();
    assert!(
        text.contains("batuta") || text.contains("Batuta") || text.contains("CLAUDE"),
        "CLAUDE.md should mention the project"
    );
}

#[test]
fn test_manifest_includes_project_instructions() {
    let m = build_default_manifest(true);
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
    let m = build_default_manifest(true);
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

// PMAT-152: exit code constants match spec §9.1
#[test]
fn test_exit_codes_match_spec() {
    assert_eq!(exit_code::SUCCESS, 0);
    assert_eq!(exit_code::AGENT_ERROR, 1);
    assert_eq!(exit_code::BUDGET_EXHAUSTED, 2);
    assert_eq!(exit_code::MAX_TURNS, 3);
    assert_eq!(exit_code::SANDBOX_VIOLATION, 4);
    assert_eq!(exit_code::NO_MODEL, 5);
}
