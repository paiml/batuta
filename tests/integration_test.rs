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

    // Create a test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let test_file = src_dir.join("main.py");
    fs::write(&test_file, "import numpy as np\nx = np.array([1, 2, 3])\n").unwrap();

    // Run analyze first to create workflow data
    Command::cargo_bin("batuta").unwrap()
        .arg("analyze")
        .arg(&src_dir)
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Now generate the report
    let report_path = temp_dir.path().join("report.html");
    Command::cargo_bin("batuta").unwrap()
        .arg("report")
        .arg("--format")
        .arg("html")
        .arg("--output")
        .arg(report_path.to_str().unwrap())
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Verify HTML structure
    assert!(report_path.exists(), "Report file should exist at {:?}", report_path);
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

// ============================================================================
// INIT COMMAND TESTS
// ============================================================================

/// Test init command with default output directory
#[test]
fn test_init_command_default() {
    let temp_dir = TempDir::new().unwrap();

    // Create test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.py"), "print('hello')").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("init")
        .arg("--source")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Initializing Batuta project"))
        .stdout(predicate::str::contains("Configuration Summary"));

    // Verify batuta.toml was created
    let config_path = temp_dir.path().join("batuta.toml");
    assert!(config_path.exists(), "batuta.toml should be created");

    // Verify configuration content
    let config_content = fs::read_to_string(&config_path).unwrap();
    assert!(config_content.contains("[project]"));
    assert!(config_content.contains("[transpilation]"));
    assert!(config_content.contains("[optimization]"));
}

/// Test init command with custom output directory
#[test]
fn test_init_command_custom_output() {
    let temp_dir = TempDir::new().unwrap();

    // Create test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("test.rs"), "fn main() {}").unwrap();

    let output_dir = temp_dir.path().join("custom-output");

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("init")
        .arg("--source")
        .arg(".")
        .arg("--output")
        .arg(&output_dir)
        .assert()
        .success();

    // Verify output directory structure
    assert!(output_dir.exists());
    assert!(output_dir.join("src").exists());

    // Verify config contains custom output path
    let config_path = temp_dir.path().join("batuta.toml");
    let config_content = fs::read_to_string(&config_path).unwrap();
    assert!(config_content.contains("custom-output"));
}

// ============================================================================
// ANALYZE COMMAND VARIANT TESTS
// ============================================================================

/// Test analyze with --tdg flag
#[test]
fn test_analyze_with_tdg() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "fn main() { println!(\"test\"); }").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--tdg")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("TDG Score"));
}

/// Test analyze with --dependencies flag
#[test]
fn test_analyze_with_dependencies() {
    let temp_dir = TempDir::new().unwrap();

    // Create a Python project with requirements.txt
    fs::write(temp_dir.path().join("requirements.txt"), "numpy>=1.20.0\npandas\n").unwrap();
    fs::write(temp_dir.path().join("main.py"), "import numpy\nimport pandas\n").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--dependencies")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Dependencies"));
}

/// Test analyze with all flags
#[test]
fn test_analyze_with_all_flags() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--tdg")
        .arg("--languages")
        .arg("--dependencies")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Analysis Results"))
        .stdout(predicate::str::contains("TDG Score"))
        .stdout(predicate::str::contains("Languages Detected"));
}

/// Test analyze with nonexistent path
#[test]
fn test_analyze_nonexistent_path() {
    Command::cargo_bin("batuta").unwrap()
        .arg("analyze")
        .arg("--languages")
        .arg("/nonexistent/path/that/does/not/exist")
        .assert()
        .failure();
}

// ============================================================================
// WORKFLOW COMMAND TESTS
// ============================================================================

/// Test transpile command without prerequisites
#[test]
fn test_transpile_without_analysis() {
    let temp_dir = TempDir::new().unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("transpile")
        .assert()
        .success()  // Should succeed but show warning
        .stdout(predicate::str::contains("Analysis phase not completed"));
}

/// Test optimize command without prerequisites
#[test]
fn test_optimize_without_transpile() {
    let temp_dir = TempDir::new().unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("optimize")
        .assert()
        .success()  // Should succeed but show warning
        .stdout(predicate::str::contains("Transpilation phase not completed"));
}

/// Test validate command without prerequisites
#[test]
fn test_validate_without_optimize() {
    let temp_dir = TempDir::new().unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("validate")
        .assert()
        .success()  // Should succeed but show warning
        .stdout(predicate::str::contains("Optimization phase not completed"));
}

/// Test build command without prerequisites
#[test]
fn test_build_without_validate() {
    let temp_dir = TempDir::new().unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("build")
        .assert()
        .success()  // Should succeed but show warning
        .stdout(predicate::str::contains("Validation phase not completed"));
}

/// Test optimize with different profiles
#[test]
fn test_optimize_profiles() {
    let profiles = vec!["fast", "balanced", "aggressive"];

    for profile in profiles {
        let temp_dir = TempDir::new().unwrap();

        // Optimize command succeeds but shows prerequisite warning
        Command::cargo_bin("batuta").unwrap()
            .current_dir(temp_dir.path())
            .arg("optimize")
            .arg("--profile")
            .arg(profile)
            .assert()
            .success()
            .stdout(predicate::str::contains("Optimizing code"));
    }
}

/// Test optimize with GPU and SIMD flags
#[test]
fn test_optimize_with_gpu_simd() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, shows warning but accepts flags
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("optimize")
        .arg("--enable-gpu")
        .arg("--enable-simd")
        .arg("--gpu-threshold")
        .arg("1000")
        .assert()
        .success()
        .stdout(predicate::str::contains("Optimizing code"));
}

/// Test validate with different flags
#[test]
fn test_validate_with_flags() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, shows warning but accepts flags
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("validate")
        .arg("--trace-syscalls")
        .arg("--diff-output")
        .arg("--run-original-tests")
        .arg("--benchmark")
        .assert()
        .success()
        .stdout(predicate::str::contains("Validating equivalence"));
}

/// Test build with different targets
#[test]
fn test_build_variants() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, all variants succeed but show warnings
    // Test release build
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--release")
        .assert()
        .success()
        .stdout(predicate::str::contains("Building Rust project"));

    // Test WASM build
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--wasm")
        .assert()
        .success()
        .stdout(predicate::str::contains("Building Rust project"));

    // Test custom target
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--target")
        .arg("x86_64-unknown-linux-gnu")
        .assert()
        .success()
        .stdout(predicate::str::contains("Building Rust project"));
}

// ============================================================================
// PARF COMMAND TESTS
// ============================================================================

/// Test parf command basic execution
#[test]
fn test_parf_command_basic() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test_function() {}").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .assert()
        .success()
        .stdout(predicate::str::contains("PARF Analysis"));
}

/// Test parf with --patterns flag
#[test]
fn test_parf_patterns() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "// TODO: implement this\npub fn test() {}").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .assert()
        .success()
        .stdout(predicate::str::contains("Code Patterns Detected"));
}

/// Test parf with --dependencies flag
#[test]
fn test_parf_dependencies() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "mod lib;\nfn main() {}").unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--dependencies")
        .assert()
        .success()
        .stdout(predicate::str::contains("Dependencies"));
}

/// Test parf with --find flag
#[test]
fn test_parf_find_references() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn my_function() {}\nfn caller() { my_function(); }").unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--find")
        .arg("my_function")
        .assert()
        .success()
        .stdout(predicate::str::contains("References"));
}

/// Test parf with different output formats
#[test]
fn test_parf_output_formats() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    // Test JSON format
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("json")
        .assert()
        .success();

    // Test Markdown format
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("markdown")
        .assert()
        .success();

    // Test text format (default)
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("text")
        .assert()
        .success();
}

/// Test parf output to file
#[test]
fn test_parf_output_file() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    let output_file = temp_dir.path().join("parf-report.txt");

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--output")
        .arg(&output_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("Report written to"));

    assert!(output_file.exists(), "PARF report file should be created");
}

// ============================================================================
// GLOBAL FLAG TESTS
// ============================================================================

/// Test --verbose flag
#[test]
fn test_verbose_flag() {
    Command::cargo_bin("batuta").unwrap()
        .arg("--verbose")
        .arg("status")
        .assert()
        .success();
}

/// Test --debug flag
#[test]
fn test_debug_flag() {
    Command::cargo_bin("batuta").unwrap()
        .arg("--debug")
        .arg("status")
        .assert()
        .success();
}

// ============================================================================
// REPORT FORMAT TESTS
// ============================================================================

/// Test report generation with text format
#[test]
fn test_report_generation_text() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.txt");

    Command::cargo_bin("batuta").unwrap()
        .arg("report")
        .arg("--format")
        .arg("text")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .success();

    assert!(report_path.exists());
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("MIGRATION REPORT"));
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

/// Test transpile with missing prerequisites
#[test]
fn test_transpile_missing_config() {
    let temp_dir = TempDir::new().unwrap();

    // Transpile without config or analysis should still succeed with warning
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("transpile")
        .assert()
        .success()
        .stdout(predicate::str::contains("Transpiling code"));
}

/// Test report generation without workflow
#[test]
fn test_report_without_workflow() {
    let temp_dir = TempDir::new().unwrap();

    let report_path = temp_dir.path().join("report.html");

    // Report command handles missing workflow data gracefully
    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("report")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("migration report"));

    // With no workflow, should either create minimal report or show warning
    // Both behaviors are acceptable
}

/// Test reset on empty workflow
#[test]
fn test_reset_empty_workflow() {
    let temp_dir = TempDir::new().unwrap();

    Command::cargo_bin("batuta").unwrap()
        .current_dir(temp_dir.path())
        .arg("reset")
        .arg("--yes")
        .assert()
        .success()
        .stdout(predicate::str::contains("No workflow state found"));
}
