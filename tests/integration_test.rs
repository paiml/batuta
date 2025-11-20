#![allow(deprecated)]

/// Integration tests for Batuta pipeline based on sovereign-ai-spec.md section 4.1 and 4.2
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Test full analysis workflow
#[test]
fn test_analyze_workflow() {
    let mut cmd = Command::cargo_bin("batuta").unwrap();

    cmd.arg("analyze")
        .arg("--languages")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Analysis Results"))
        .stdout(predicate::str::contains("Rust"));
}

/// Test workflow state persistence
#[test]
fn test_workflow_state_tracking() {
    let temp_dir = TempDir::new().unwrap();

    // Create a small test project structure
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "fn main() { println!(\"test\"); }").unwrap();

    let state_file = temp_dir.path().join(".batuta-state.json");

    // Run analyze to create state
    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--languages")
        .arg(".")
        .assert()
        .success();

    // Verify state file exists
    assert!(state_file.exists(), "State file should be created");

    // Read and verify state structure
    let state_content = fs::read_to_string(&state_file).unwrap();
    assert!(state_content.contains("Analysis"));
    assert!(state_content.contains("Completed"));
}

/// Test status command
#[test]
fn test_status_command() {
    let mut cmd = Command::cargo_bin("batuta").unwrap();

    cmd.arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Workflow Status"));
}

/// Test report generation (markdown)
#[test]
fn test_report_generation_markdown() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.md");

    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.arg("report")
        .arg("--format")
        .arg("markdown")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Report generated successfully"));

    // Verify report file exists and has content
    assert!(report_path.exists(), "Report file should be created");
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("# Migration Report"));
    assert!(content.contains("## Summary"));
}

/// Test report generation (JSON)
#[test]
fn test_report_generation_json() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.json");

    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.arg("report")
        .arg("--format")
        .arg("json")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .success();

    // Verify JSON is valid
    assert!(report_path.exists());
    let content = fs::read_to_string(&report_path).unwrap();
    let _: serde_json::Value = serde_json::from_str(&content)
        .expect("Report should be valid JSON");
}

/// Test report generation (HTML)
#[test]
fn test_report_generation_html() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.html");

    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.arg("report")
        .arg("--format")
        .arg("html")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .success();

    // Verify HTML structure
    assert!(report_path.exists());
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("<html"));
    assert!(content.contains("Migration Report"));
}

/// Test reset workflow
#[test]
fn test_reset_workflow() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    // First run analyze to create state
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg(".")
        .assert()
        .success();

    // Then reset with --yes flag
    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.current_dir(temp_dir.path())
        .arg("reset")
        .arg("--yes")
        .assert()
        .success()
        .stdout(predicate::str::contains("reset successfully"));
}

/// Test help command
#[test]
fn test_help_command() {
    let mut cmd = Command::cargo_bin("batuta").unwrap();

    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Orchestration framework"))
        .stdout(predicate::str::contains("analyze"))
        .stdout(predicate::str::contains("transpile"))
        .stdout(predicate::str::contains("optimize"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("build"));
}

/// Test CLI version
#[test]
fn test_version_command() {
    let mut cmd = Command::cargo_bin("batuta").unwrap();

    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("batuta"));
}

/// Test Renacer syscall tracing validation (BATUTA-011)
#[test]
fn test_renacer_validation() {
    use batuta::pipeline::{PipelineContext, ValidationStage, PipelineStage};
    use std::path::PathBuf;
    use tokio::runtime::Runtime;

    let temp_dir = TempDir::new().unwrap();

    // Create a simple test context
    let ctx = PipelineContext::new(
        PathBuf::from(temp_dir.path()),
        PathBuf::from(temp_dir.path()),
    );

    // Create ValidationStage with syscall tracing enabled
    let stage = ValidationStage::new(true, false);

    // Run validation (will skip if binaries don't exist)
    let rt = Runtime::new().unwrap();
    let result = rt.block_on(stage.execute(ctx));

    // Should succeed even if binaries don't exist (graceful handling)
    assert!(result.is_ok(), "ValidationStage should handle missing binaries gracefully");

    let final_ctx = result.unwrap();
    assert!(
        final_ctx.metadata.contains_key("validation_completed"),
        "Validation should mark completion in metadata"
    );
}
