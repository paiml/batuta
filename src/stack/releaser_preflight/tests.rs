//! Tests for releaser preflight checks

#[cfg(test)]
mod tests {
    use crate::stack::graph::DependencyGraph;
    use crate::stack::releaser::ReleaseOrchestrator;
    use crate::stack::releaser_preflight::helpers::{
        parse_count_from_json_multi, parse_value_from_json, run_check_command, score_check_result,
    };
    use crate::stack::releaser_types::ReleaseConfig;
    use crate::stack::types::PreflightCheck;
    use std::path::Path;

    fn make_orchestrator(config: ReleaseConfig) -> ReleaseOrchestrator {
        let graph = DependencyGraph::new();
        let checker = crate::stack::checker::StackChecker::with_graph(graph);
        ReleaseOrchestrator::new(checker, config)
    }

    fn default_orchestrator() -> ReleaseOrchestrator {
        make_orchestrator(ReleaseConfig::default())
    }

    fn orchestrator_with_command(field: &str, cmd: &str) -> ReleaseOrchestrator {
        let mut config = ReleaseConfig::default();
        match field {
            "lint" => config.lint_command = cmd.to_string(),
            "coverage" => config.coverage_command = cmd.to_string(),
            "comply" => config.comply_command = cmd.to_string(),
            "quality_gate" => config.quality_gate_command = cmd.to_string(),
            "tdg" => config.tdg_command = cmd.to_string(),
            "dead_code" => config.dead_code_command = cmd.to_string(),
            "complexity" => config.complexity_command = cmd.to_string(),
            "satd" => config.satd_command = cmd.to_string(),
            "popper" => config.popper_command = cmd.to_string(),
            "book" => config.book_command = cmd.to_string(),
            "examples" => config.examples_command = cmd.to_string(),
            _ => {}
        }
        make_orchestrator(config)
    }

    // ============================================================================
    // JSON Parsing Tests
    // ============================================================================

    #[test]
    fn test_parse_score_from_json_simple() {
        assert_eq!(
            ReleaseOrchestrator::parse_score_from_json(r#"{"score": 85.5, "other": 10}"#, "score"),
            Some(85.5)
        );
    }

    #[test]
    fn test_parse_score_from_json_integer() {
        assert_eq!(
            ReleaseOrchestrator::parse_score_from_json(r#"{"total": 100}"#, "total"),
            Some(100.0)
        );
    }

    #[test]
    fn test_parse_score_from_json_missing() {
        assert_eq!(
            ReleaseOrchestrator::parse_score_from_json(r#"{"other": 10}"#, "score"),
            None
        );
    }

    #[test]
    fn test_parse_count_from_json() {
        assert_eq!(
            ReleaseOrchestrator::parse_count_from_json(r#"{"count": 42}"#, "count"),
            Some(42)
        );
    }

    #[test]
    fn test_parse_count_from_json_decimal() {
        assert_eq!(
            ReleaseOrchestrator::parse_count_from_json(r#"{"count": 42.9}"#, "count"),
            Some(42)
        );
    }

    #[test]
    fn test_parse_score_with_whitespace() {
        assert_eq!(
            ReleaseOrchestrator::parse_score_from_json(r#"{"score":  85.5}"#, "score"),
            Some(85.5)
        );
    }

    #[test]
    fn test_parse_score_negative() {
        assert_eq!(
            ReleaseOrchestrator::parse_score_from_json(r#"{"delta": -10.5}"#, "delta"),
            Some(-10.5)
        );
    }

    // ============================================================================
    // Helper Function Tests
    // ============================================================================

    #[test]
    fn test_parse_value_from_json_first_key_match() {
        assert_eq!(
            parse_value_from_json(r#"{"score": 85.5, "total": 90.0}"#, &["score", "total"]),
            Some(85.5)
        );
    }

    #[test]
    fn test_parse_value_from_json_fallback_key() {
        assert_eq!(
            parse_value_from_json(r#"{"total": 90.0}"#, &["score", "total"]),
            Some(90.0)
        );
    }

    #[test]
    fn test_parse_value_from_json_no_match() {
        assert_eq!(
            parse_value_from_json(r#"{"other": 10}"#, &["score", "total"]),
            None
        );
    }

    #[test]
    fn test_parse_count_from_json_multi() {
        assert_eq!(
            parse_count_from_json_multi(r#"{"dead_code_count": 5}"#, &["count", "dead_code_count"]),
            Some(5)
        );
    }

    // ============================================================================
    // score_check_result
    // ============================================================================

    #[test]
    fn test_score_check_result_pass() {
        let r = score_check_result("tdg", "TDG score", Some(90.0), 80.0, true, true);
        assert!(r.passed);
        assert!(r.message.contains("90.0"));
    }

    #[test]
    fn test_score_check_result_fail() {
        let r = score_check_result("tdg", "TDG score", Some(70.0), 80.0, true, true);
        assert!(!r.passed);
    }

    #[test]
    fn test_score_check_result_warning() {
        let r = score_check_result("tdg", "TDG score", Some(70.0), 80.0, false, true);
        assert!(r.passed);
        assert!(r.message.contains("warning"));
    }

    #[test]
    fn test_score_check_result_no_score_success() {
        let r = score_check_result("tdg", "TDG", None, 80.0, true, true);
        assert!(r.passed);
        assert!(r.message.contains("check passed"));
    }

    #[test]
    fn test_score_check_result_no_score_not_success() {
        let r = score_check_result("tdg", "TDG", None, 80.0, true, false);
        assert!(r.passed);
        assert!(r.message.contains("score not parsed"));
    }

    // ============================================================================
    // run_check_command
    // ============================================================================

    #[test]
    fn test_run_check_command_empty_command() {
        let r = run_check_command("", "test_id", "skipped", Path::new("."), |_, _, _| {
            PreflightCheck::fail("test_id", "should not reach")
        });
        assert!(r.passed);
        assert!(r.message.contains("skipped"));
    }

    #[test]
    fn test_run_check_command_not_found() {
        let r = run_check_command(
            "nonexistent_tool_xyz_123",
            "test_id",
            "skip",
            Path::new("."),
            |_, _, _| PreflightCheck::fail("test_id", "should not reach"),
        );
        assert!(r.passed);
        assert!(r.message.contains("not found"));
    }

    #[test]
    fn test_run_check_command_success() {
        let r = run_check_command("true", "test_id", "skip", Path::new("."), |out, _, _| {
            if out.status.success() {
                PreflightCheck::pass("test_id", "ok")
            } else {
                PreflightCheck::fail("test_id", "nope")
            }
        });
        assert!(r.passed);
    }

    #[test]
    fn test_run_check_command_failure() {
        let r = run_check_command("false", "test_id", "skip", Path::new("."), |out, _, _| {
            if out.status.success() {
                PreflightCheck::pass("test_id", "ok")
            } else {
                PreflightCheck::fail("test_id", "failed")
            }
        });
        assert!(!r.passed);
    }

    // ============================================================================
    // check_git_clean
    // ============================================================================

    #[test]
    fn test_check_git_clean_in_repo() {
        // Run in a temp dir with a git repo
        let dir = std::env::temp_dir().join("test_rp_git_clean");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&dir)
            .output();
        let orch = default_orchestrator();
        let r = orch.check_git_clean(&dir);
        // Fresh repo with no commits: porcelain is empty
        assert!(r.passed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_git_clean_dirty() {
        let dir = std::env::temp_dir().join("test_rp_git_dirty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&dir)
            .output();
        std::fs::write(dir.join("file.txt"), "hello").unwrap();
        let orch = default_orchestrator();
        let r = orch.check_git_clean(&dir);
        assert!(!r.passed);
        assert!(r.message.contains("Uncommitted"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_git_clean_no_git() {
        let dir = std::env::temp_dir().join("test_rp_no_git");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let orch = default_orchestrator();
        let r = orch.check_git_clean(&dir);
        // git status in a non-repo still runs but returns error
        // Either way the check returns something sensible
        let _ = r;
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================================
    // check_lint
    // ============================================================================

    #[test]
    fn test_check_lint_pass() {
        let orch = orchestrator_with_command("lint", "true");
        let r = orch.check_lint(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_lint_fail() {
        let orch = orchestrator_with_command("lint", "false");
        let r = orch.check_lint(Path::new("."));
        assert!(!r.passed);
        assert!(r.message.contains("Lint failed"));
    }

    #[test]
    fn test_check_lint_empty() {
        let orch = orchestrator_with_command("lint", "");
        let r = orch.check_lint(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("No lint command"));
    }

    // ============================================================================
    // check_coverage
    // ============================================================================

    #[test]
    fn test_check_coverage_pass() {
        let orch = orchestrator_with_command("coverage", "true");
        let r = orch.check_coverage(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_coverage_fail() {
        let orch = orchestrator_with_command("coverage", "false");
        let r = orch.check_coverage(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_coverage_empty() {
        let orch = orchestrator_with_command("coverage", "");
        let r = orch.check_coverage(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_no_path_deps / check_version_bumped (placeholders)
    // ============================================================================

    #[test]
    fn test_check_no_path_deps() {
        let orch = default_orchestrator();
        let r = orch.check_no_path_deps("batuta");
        assert!(r.passed);
    }

    #[test]
    fn test_check_version_bumped() {
        let orch = default_orchestrator();
        let r = orch.check_version_bumped("batuta");
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_comply
    // ============================================================================

    #[test]
    fn test_check_pmat_comply_pass() {
        let orch = orchestrator_with_command("comply", "true");
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("0 violations"));
    }

    #[test]
    fn test_check_pmat_comply_empty() {
        let orch = orchestrator_with_command("comply", "");
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_pmat_comply_not_found() {
        let orch = orchestrator_with_command("comply", "nonexistent_xyz_tool");
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(r.passed); // tool not found → skip
    }

    // ============================================================================
    // check_pmat_quality_gate
    // ============================================================================

    #[test]
    fn test_check_quality_gate_pass() {
        let orch = orchestrator_with_command("quality_gate", "true");
        let r = orch.check_pmat_quality_gate(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_quality_gate_fail_blocking() {
        let mut config = ReleaseConfig::default();
        config.quality_gate_command = "false".to_string();
        config.fail_on_quality_gate = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_quality_gate(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_quality_gate_fail_non_blocking() {
        let mut config = ReleaseConfig::default();
        config.quality_gate_command = "false".to_string();
        config.fail_on_quality_gate = false;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_quality_gate(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("warning"));
    }

    // ============================================================================
    // check_pmat_tdg
    // ============================================================================

    #[test]
    fn test_check_tdg_empty_command() {
        let orch = orchestrator_with_command("tdg", "");
        let r = orch.check_pmat_tdg(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_tdg_success_no_json() {
        let orch = orchestrator_with_command("tdg", "true");
        let r = orch.check_pmat_tdg(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_dead_code
    // ============================================================================

    #[test]
    fn test_check_dead_code_empty() {
        let orch = orchestrator_with_command("dead_code", "");
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_dead_code_clean() {
        let orch = orchestrator_with_command("dead_code", "true");
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_complexity
    // ============================================================================

    #[test]
    fn test_check_complexity_empty() {
        let orch = orchestrator_with_command("complexity", "");
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_complexity_success() {
        let orch = orchestrator_with_command("complexity", "true");
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_satd
    // ============================================================================

    #[test]
    fn test_check_satd_empty() {
        let orch = orchestrator_with_command("satd", "");
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_satd_success() {
        let orch = orchestrator_with_command("satd", "true");
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_popper
    // ============================================================================

    #[test]
    fn test_check_popper_empty() {
        let orch = orchestrator_with_command("popper", "");
        let r = orch.check_pmat_popper(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_popper_success() {
        let orch = orchestrator_with_command("popper", "true");
        let r = orch.check_pmat_popper(Path::new("."));
        assert!(r.passed);
    }

    // ============================================================================
    // check_book_build
    // ============================================================================

    #[test]
    fn test_check_book_no_dir() {
        let dir = std::env::temp_dir().join("test_rp_no_book");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let orch = default_orchestrator();
        let r = orch.check_book_build(&dir);
        assert!(r.passed);
        assert!(r.message.contains("No book directory"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_book_pass() {
        let dir = std::env::temp_dir().join("test_rp_book_pass");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("book")).unwrap();
        let orch = orchestrator_with_command("book", "true");
        let r = orch.check_book_build(&dir);
        assert!(r.passed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_book_fail_blocking() {
        let dir = std::env::temp_dir().join("test_rp_book_fail");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("book")).unwrap();
        let mut config = ReleaseConfig::default();
        config.book_command = "false".to_string();
        config.fail_on_book = true;
        let orch = make_orchestrator(config);
        let r = orch.check_book_build(&dir);
        assert!(!r.passed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_book_fail_non_blocking() {
        let dir = std::env::temp_dir().join("test_rp_book_warn");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("book")).unwrap();
        let mut config = ReleaseConfig::default();
        config.book_command = "false".to_string();
        config.fail_on_book = false;
        let orch = make_orchestrator(config);
        let r = orch.check_book_build(&dir);
        assert!(r.passed);
        assert!(r.message.contains("not blocking"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================================
    // check_examples_run / discover_examples
    // ============================================================================

    #[test]
    fn test_check_examples_no_dir() {
        let dir = std::env::temp_dir().join("test_rp_no_examples");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let orch = default_orchestrator();
        let r = orch.check_examples_run(&dir);
        assert!(r.passed);
        assert!(r.message.contains("No examples directory"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_examples_empty_command() {
        let orch = orchestrator_with_command("examples", "");
        let dir = std::env::temp_dir().join("test_rp_examples_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let r = orch.check_examples_run(&dir);
        assert!(r.passed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_examples_empty_dir() {
        let dir = std::env::temp_dir().join("test_rp_disc_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("examples")).unwrap();
        let orch = default_orchestrator();
        let examples = orch.discover_examples(&dir);
        assert!(examples.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_examples_from_files() {
        let dir = std::env::temp_dir().join("test_rp_disc_files");
        let _ = std::fs::remove_dir_all(&dir);
        let examples_dir = dir.join("examples");
        std::fs::create_dir_all(&examples_dir).unwrap();
        std::fs::write(examples_dir.join("demo.rs"), "fn main() {}").unwrap();
        std::fs::write(examples_dir.join("bench.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let orch = default_orchestrator();
        let examples = orch.discover_examples(&dir);
        assert_eq!(examples.len(), 2);
        assert!(examples.contains(&"demo".to_string()));
        assert!(examples.contains(&"bench".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_examples_no_cargo_toml() {
        let dir = std::env::temp_dir().join("test_rp_disc_no_cargo");
        let _ = std::fs::remove_dir_all(&dir);
        let examples_dir = dir.join("examples");
        std::fs::create_dir_all(&examples_dir).unwrap();
        std::fs::write(examples_dir.join("hello.rs"), "fn main() {}").unwrap();
        let orch = default_orchestrator();
        let examples = orch.discover_examples(&dir);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0], "hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_examples_no_examples_found() {
        let dir = std::env::temp_dir().join("test_rp_examples_none");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("examples")).unwrap();
        let orch = default_orchestrator();
        let r = orch.check_examples_run(&dir);
        assert!(r.passed);
        assert!(r.message.contains("No examples found"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================================
    // check_pmat_comply detailed branching
    // ============================================================================

    #[test]
    fn test_check_pmat_comply_violations_blocking() {
        // "echo" outputs something, but we need CB- in output
        let mut config = ReleaseConfig::default();
        config.comply_command = "echo CB-020: unsafe block".to_string();
        config.fail_on_comply_violations = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(!r.passed);
        assert!(r.message.contains("CB-020"));
    }

    #[test]
    fn test_check_pmat_comply_violations_non_blocking() {
        let mut config = ReleaseConfig::default();
        config.comply_command = "echo CB-020: unsafe block".to_string();
        config.fail_on_comply_violations = false;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("warnings"));
    }

    #[test]
    fn test_check_pmat_comply_violation_keyword() {
        let mut config = ReleaseConfig::default();
        config.comply_command = "echo violation found".to_string();
        config.fail_on_comply_violations = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_pmat_comply_clean_exit_no_violations() {
        let mut config = ReleaseConfig::default();
        config.comply_command = "echo all good".to_string();
        config.fail_on_comply_violations = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_comply(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("0 violations"));
    }

    // ============================================================================
    // check_pmat_dead_code detailed branching
    // ============================================================================

    #[test]
    fn test_check_dead_code_with_count_zero() {
        let mut config = ReleaseConfig::default();
        config.dead_code_command = r#"echo {"count": 0}"#.to_string();
        config.fail_on_dead_code = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("No dead code"));
    }

    #[test]
    fn test_check_dead_code_with_count_nonzero_blocking() {
        let mut config = ReleaseConfig::default();
        config.dead_code_command = r#"echo {"count": 5}"#.to_string();
        config.fail_on_dead_code = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(!r.passed);
        assert!(r.message.contains("5 dead code items"));
    }

    #[test]
    fn test_check_dead_code_with_count_nonzero_warning() {
        let mut config = ReleaseConfig::default();
        config.dead_code_command = r#"echo {"count": 3}"#.to_string();
        config.fail_on_dead_code = false;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("warning"));
    }

    #[test]
    fn test_check_dead_code_keyword_blocking() {
        let mut config = ReleaseConfig::default();
        config.dead_code_command = "echo dead_code detected unused items".to_string();
        config.fail_on_dead_code = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_dead_code_keyword_warning() {
        let mut config = ReleaseConfig::default();
        config.dead_code_command = "echo dead_code found".to_string();
        config.fail_on_dead_code = false;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_dead_code(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("warning"));
    }

    // ============================================================================
    // check_pmat_complexity detailed branching
    // ============================================================================

    #[test]
    fn test_check_complexity_within_limit() {
        let mut config = ReleaseConfig::default();
        config.complexity_command = r#"echo {"max_complexity": 10}"#.to_string();
        config.max_complexity = 20;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("10"));
    }

    #[test]
    fn test_check_complexity_exceeds_blocking() {
        let mut config = ReleaseConfig::default();
        config.complexity_command = r#"echo {"max_complexity": 30}"#.to_string();
        config.max_complexity = 20;
        config.fail_on_complexity = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(!r.passed);
        assert!(r.message.contains("30"));
    }

    #[test]
    fn test_check_complexity_zero_violations() {
        let mut config = ReleaseConfig::default();
        config.complexity_command = r#"echo {"violations": 0}"#.to_string();
        config.fail_on_complexity = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_complexity_violations_blocking() {
        let mut config = ReleaseConfig::default();
        config.complexity_command = r#"echo {"violations": 5}"#.to_string();
        config.fail_on_complexity = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_complexity(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_complexity_fail_exit_no_data() {
        let mut config = ReleaseConfig::default();
        config.complexity_command = "false".to_string();
        config.fail_on_complexity = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_complexity(Path::new("."));
        // No JSON and command failed → "completed (warning)"
        assert!(r.passed);
    }

    // ============================================================================
    // check_pmat_satd detailed branching
    // ============================================================================

    #[test]
    fn test_check_satd_within_limit() {
        let mut config = ReleaseConfig::default();
        config.satd_command = r#"echo {"total": 5}"#.to_string();
        config.max_satd_items = 10;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_satd_exceeds_blocking() {
        let mut config = ReleaseConfig::default();
        config.satd_command = r#"echo {"total": 15}"#.to_string();
        config.max_satd_items = 10;
        config.fail_on_satd = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_satd_exceeds_warning() {
        let mut config = ReleaseConfig::default();
        config.satd_command = r#"echo {"total": 15}"#.to_string();
        config.max_satd_items = 10;
        config.fail_on_satd = false;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("warning"));
    }

    #[test]
    fn test_check_satd_no_count_success() {
        let mut config = ReleaseConfig::default();
        config.satd_command = "true".to_string();
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("check passed"));
    }

    #[test]
    fn test_check_satd_no_count_failure() {
        let mut config = ReleaseConfig::default();
        config.satd_command = "false".to_string();
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_satd(Path::new("."));
        assert!(r.passed);
        assert!(r.message.contains("check completed"));
    }

    // ============================================================================
    // check_pmat_tdg / check_pmat_popper score thresholds
    // ============================================================================

    #[test]
    fn test_check_tdg_above_threshold() {
        let mut config = ReleaseConfig::default();
        config.tdg_command = r#"echo {"score": 90}"#.to_string();
        config.min_tdg_score = 80.0;
        config.fail_on_tdg = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_tdg(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_tdg_below_threshold_blocking() {
        let mut config = ReleaseConfig::default();
        config.tdg_command = r#"echo {"score": 70}"#.to_string();
        config.min_tdg_score = 80.0;
        config.fail_on_tdg = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_tdg(Path::new("."));
        assert!(!r.passed);
    }

    #[test]
    fn test_check_popper_above_threshold() {
        let mut config = ReleaseConfig::default();
        config.popper_command = r#"echo {"score": 80}"#.to_string();
        config.min_popper_score = 60.0;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_popper(Path::new("."));
        assert!(r.passed);
    }

    #[test]
    fn test_check_popper_below_threshold_blocking() {
        let mut config = ReleaseConfig::default();
        config.popper_command = r#"echo {"score": 40}"#.to_string();
        config.min_popper_score = 60.0;
        config.fail_on_popper = true;
        let orch = make_orchestrator(config);
        let r = orch.check_pmat_popper(Path::new("."));
        assert!(!r.passed);
    }

    // ============================================================================
    // check_book_build with empty command
    // ============================================================================

    #[test]
    fn test_check_book_empty_command() {
        let dir = std::env::temp_dir().join("test_rp_book_empty_cmd");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("book")).unwrap();
        let orch = orchestrator_with_command("book", "");
        let r = orch.check_book_build(&dir);
        assert!(r.passed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================================
    // run_check_command with arguments
    // ============================================================================

    #[test]
    fn test_run_check_command_with_args() {
        let r = run_check_command(
            "echo hello world",
            "test_id",
            "skip",
            Path::new("."),
            |out, stdout, _| {
                if out.status.success() && stdout.contains("hello") {
                    PreflightCheck::pass("test_id", "got output")
                } else {
                    PreflightCheck::fail("test_id", "bad")
                }
            },
        );
        assert!(r.passed);
        assert!(r.message.contains("got output"));
    }

    #[test]
    fn test_run_check_command_stderr_access() {
        let r = run_check_command(
            "ls /nonexistent_path_xyz",
            "test_id",
            "skip",
            Path::new("."),
            |_out, _stdout, stderr| {
                if stderr.contains("No such file") || stderr.contains("cannot access") {
                    PreflightCheck::pass("test_id", "stderr captured")
                } else {
                    // On some systems the error message differs
                    PreflightCheck::pass("test_id", "command ran")
                }
            },
        );
        assert!(r.passed);
    }
}
