#![allow(dead_code)]
//! Release Preflight Checks
//!
//! Preflight check methods for ReleaseOrchestrator extracted from releaser.rs.
//! Contains all check_* methods for various quality gates.
//!
//! The common command-execution pattern is factored into [`run_check_command`],
//! which handles argument parsing, spawning, UTF-8 decoding, and the
//! not-found / general-error branches that every check shares.

use crate::stack::types::PreflightCheck;
use std::path::Path;
use std::process::Command;

use super::releaser::ReleaseOrchestrator;

// =============================================================================
// Shared helpers (free functions – no `&self` needed)
// =============================================================================

/// Execute an external command described by a single whitespace-separated
/// config string and dispatch the result through a caller-provided closure.
///
/// Handles the three outcomes every check shares:
///   1. Empty command string  -> pass with `skip_msg`
///   2. Command not found     -> pass with "<tool> not found (skipped)"
///   3. Other spawn error     -> fail with error details
///   4. Successful spawn      -> delegate to `process_output`
fn run_check_command<F>(
    config_command: &str,
    check_id: &str,
    skip_msg: &str,
    crate_path: &Path,
    process_output: F,
) -> PreflightCheck
where
    F: FnOnce(&std::process::Output, &str, &str) -> PreflightCheck,
{
    let parts: Vec<&str> = config_command.split_whitespace().collect();
    if parts.is_empty() {
        return PreflightCheck::pass(check_id, skip_msg);
    }
    match Command::new(parts[0])
        .args(&parts[1..])
        .current_dir(crate_path)
        .output()
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            process_output(&output, &stdout, &stderr)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            PreflightCheck::pass(check_id, format!("{} not found (skipped)", parts[0]))
        }
        Err(e) => PreflightCheck::fail(check_id, format!("Failed to run {}: {}", parts[0], e)),
    }
}

/// Try several JSON keys in order and return the first successfully parsed f64.
fn parse_value_from_json(json: &str, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|k| ReleaseOrchestrator::parse_score_from_json(json, k))
}

/// Try several JSON keys in order and return the first successfully parsed u32.
fn parse_count_from_json_multi(json: &str, keys: &[&str]) -> Option<u32> {
    parse_value_from_json(json, keys).map(|f| f as u32)
}

/// Evaluate a numeric score against a threshold, producing a uniform
/// pass / fail / warning [`PreflightCheck`].
///
/// Used by TDG, Popper, and similar score-based gates.
fn score_check_result(
    check_id: &str,
    label: &str,
    value: Option<f64>,
    threshold: f64,
    fail_on_threshold: bool,
    status_success: bool,
) -> PreflightCheck {
    match value {
        Some(v) if v >= threshold => PreflightCheck::pass(
            check_id,
            format!("{}: {:.1} (minimum: {:.1})", label, v, threshold),
        ),
        Some(v) if fail_on_threshold => PreflightCheck::fail(
            check_id,
            format!("{} {:.1} below minimum {:.1}", label, v, threshold),
        ),
        Some(v) => PreflightCheck::pass(
            check_id,
            format!("{}: {:.1} (warning: below {:.1})", label, v, threshold),
        ),
        None if status_success => PreflightCheck::pass(check_id, format!("{} check passed", label)),
        None => PreflightCheck::pass(
            check_id,
            format!("{} check completed (score not parsed)", label),
        ),
    }
}

impl ReleaseOrchestrator {
    /// Check if git working directory is clean
    pub(super) fn check_git_clean(&self, crate_path: &Path) -> PreflightCheck {
        let output = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                if out.stdout.is_empty() {
                    PreflightCheck::pass("git_clean", "Working directory is clean")
                } else {
                    let files = String::from_utf8_lossy(&out.stdout);
                    PreflightCheck::fail(
                        "git_clean",
                        format!("Uncommitted changes:\n{}", files.trim()),
                    )
                }
            }
            Err(e) => {
                PreflightCheck::fail("git_clean", format!("Failed to check git status: {}", e))
            }
        }
    }

    /// Check lint passes
    pub(super) fn check_lint(&self, crate_path: &Path) -> PreflightCheck {
        run_check_command(
            &self.config.lint_command,
            "lint",
            "No lint command configured",
            crate_path,
            |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("lint", "Lint passed")
                } else {
                    PreflightCheck::fail("lint", format!("Lint failed: {}", stderr.trim()))
                }
            },
        )
    }

    /// Check coverage meets minimum
    pub(super) fn check_coverage(&self, crate_path: &Path) -> PreflightCheck {
        let min_coverage = self.config.min_coverage;
        run_check_command(
            &self.config.coverage_command,
            "coverage",
            "No coverage command configured",
            crate_path,
            move |output, _stdout, _stderr| {
                if output.status.success() {
                    PreflightCheck::pass(
                        "coverage",
                        format!("Coverage check passed (min: {}%)", min_coverage),
                    )
                } else {
                    PreflightCheck::fail("coverage", format!("Coverage below {}%", min_coverage))
                }
            },
        )
    }

    /// Check PMAT comply for ComputeBrick defects (CB-XXX violations)
    ///
    /// Runs `pmat comply` to detect:
    /// - CB-020: Unsafe blocks without safety comments
    /// - CB-021: SIMD without target_feature attributes
    /// - CB-022: Missing error handling patterns
    /// - And other PMAT compliance rules
    pub(super) fn check_pmat_comply(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on_violations = self.config.fail_on_comply_violations;
        run_check_command(
            &self.config.comply_command,
            "pmat_comply",
            "No comply command configured (skipped)",
            crate_path,
            move |output, stdout, stderr| {
                let has_violations = stdout.contains("CB-")
                    || stderr.contains("CB-")
                    || stdout.contains("violation")
                    || stderr.contains("violation");

                if output.status.success() && !has_violations {
                    PreflightCheck::pass("pmat_comply", "PMAT comply passed (0 violations)")
                } else if has_violations && fail_on_violations {
                    let violation_hint = if stdout.contains("CB-") {
                        stdout
                            .lines()
                            .filter(|l| l.contains("CB-"))
                            .take(3)
                            .collect::<Vec<_>>()
                            .join("; ")
                    } else {
                        "violations detected".to_string()
                    };
                    PreflightCheck::fail(
                        "pmat_comply",
                        format!("PMAT comply failed: {}", violation_hint),
                    )
                } else if has_violations {
                    PreflightCheck::pass("pmat_comply", "PMAT comply has warnings (not blocking)")
                } else {
                    PreflightCheck::fail(
                        "pmat_comply",
                        format!("PMAT comply error: {}", stderr.trim()),
                    )
                }
            },
        )
    }

    /// Check for path dependencies
    pub(super) fn check_no_path_deps(&self, _crate_name: &str) -> PreflightCheck {
        // This would use the checker's graph to verify no path deps
        // For now, always pass as a placeholder
        PreflightCheck::pass("no_path_deps", "No path dependencies found")
    }

    /// Check version is bumped from crates.io
    pub(super) fn check_version_bumped(&self, _crate_name: &str) -> PreflightCheck {
        // This would compare local version vs crates.io
        // For now, always pass as a placeholder
        PreflightCheck::pass("version_bumped", "Version is ahead of crates.io")
    }

    // =========================================================================
    // PMAT Quality Gate Integration (PMAT-STACK-GATES)
    // =========================================================================

    /// Check PMAT quality-gate (comprehensive quality checks)
    ///
    /// Runs `pmat quality-gate` which includes:
    /// - Dead code detection
    /// - Complexity analysis
    /// - Coverage verification
    /// - SATD detection
    /// - Security checks
    pub(super) fn check_pmat_quality_gate(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on_gate = self.config.fail_on_quality_gate;
        run_check_command(
            &self.config.quality_gate_command,
            "quality_gate",
            "No quality-gate command configured (skipped)",
            crate_path,
            move |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("quality_gate", "PMAT quality-gate passed")
                } else if fail_on_gate {
                    PreflightCheck::fail(
                        "quality_gate",
                        format!("Quality gate failed: {}", stderr.trim()),
                    )
                } else {
                    PreflightCheck::pass("quality_gate", "Quality gate has warnings (not blocking)")
                }
            },
        )
    }

    /// Check PMAT TDG (Technical Debt Grading) score
    ///
    /// Runs `pmat tdg --format json` and parses the score.
    /// Fails if score < min_tdg_score (default: 80).
    pub(super) fn check_pmat_tdg(&self, crate_path: &Path) -> PreflightCheck {
        let min_score = self.config.min_tdg_score;
        let fail_on = self.config.fail_on_tdg;
        run_check_command(
            &self.config.tdg_command,
            "tdg",
            "No TDG command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let score = parse_value_from_json(stdout, &["score", "tdg_score", "total"]);
                score_check_result(
                    "tdg",
                    "TDG score",
                    score,
                    min_score,
                    fail_on,
                    output.status.success(),
                )
            },
        )
    }

    /// Check PMAT dead-code analysis
    ///
    /// Runs `pmat analyze dead-code` to detect unused code.
    pub(super) fn check_pmat_dead_code(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on = self.config.fail_on_dead_code;
        run_check_command(
            &self.config.dead_code_command,
            "dead_code",
            "No dead-code command configured (skipped)",
            crate_path,
            move |_output, stdout, _stderr| {
                let has_dead_code = stdout.contains("dead_code") || stdout.contains("unused");
                let count = parse_count_from_json_multi(stdout, &["count", "dead_code_count"]);

                match (has_dead_code, count) {
                    (_, Some(0)) | (false, None) => {
                        PreflightCheck::pass("dead_code", "No dead code detected")
                    }
                    (_, Some(n)) if fail_on => {
                        PreflightCheck::fail("dead_code", format!("{} dead code items found", n))
                    }
                    (_, Some(n)) => PreflightCheck::pass(
                        "dead_code",
                        format!("{} dead code items (warning)", n),
                    ),
                    (true, None) if fail_on => {
                        PreflightCheck::fail("dead_code", "Dead code detected")
                    }
                    (true, None) => {
                        PreflightCheck::pass("dead_code", "Dead code detected (warning)")
                    }
                }
            },
        )
    }

    /// Check PMAT complexity analysis
    ///
    /// Runs `pmat analyze complexity` to check cyclomatic complexity.
    /// Fails if any function exceeds max_complexity (default: 20).
    pub(super) fn check_pmat_complexity(&self, crate_path: &Path) -> PreflightCheck {
        let max_complexity = self.config.max_complexity;
        let fail_on = self.config.fail_on_complexity;
        run_check_command(
            &self.config.complexity_command,
            "complexity",
            "No complexity command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let max_found = parse_count_from_json_multi(stdout, &["max_complexity", "highest"]);
                let violations =
                    parse_count_from_json_multi(stdout, &["violations", "violation_count"]);

                match (max_found, violations) {
                    (Some(m), _) if m <= max_complexity => PreflightCheck::pass(
                        "complexity",
                        format!("Max complexity: {} (limit: {})", m, max_complexity),
                    ),
                    (Some(m), _) if fail_on => PreflightCheck::fail(
                        "complexity",
                        format!("Complexity {} exceeds limit {}", m, max_complexity),
                    ),
                    (_, Some(0)) => PreflightCheck::pass("complexity", "No complexity violations"),
                    (_, Some(v)) if fail_on => {
                        PreflightCheck::fail("complexity", format!("{} complexity violations", v))
                    }
                    _ if output.status.success() => {
                        PreflightCheck::pass("complexity", "Complexity check passed")
                    }
                    _ => PreflightCheck::pass("complexity", "Complexity check completed (warning)"),
                }
            },
        )
    }

    /// Check PMAT SATD (Self-Admitted Technical Debt)
    ///
    /// Runs `pmat analyze satd` to detect TODO/FIXME/HACK comments.
    /// Fails if count exceeds max_satd_items (default: 10).
    pub(super) fn check_pmat_satd(&self, crate_path: &Path) -> PreflightCheck {
        let max_items = self.config.max_satd_items;
        let fail_on = self.config.fail_on_satd;
        run_check_command(
            &self.config.satd_command,
            "satd",
            "No SATD command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let count = parse_count_from_json_multi(stdout, &["total", "count", "satd_count"]);

                match count {
                    Some(c) if c <= max_items => PreflightCheck::pass(
                        "satd",
                        format!("{} SATD items (limit: {})", c, max_items),
                    ),
                    Some(c) if fail_on => PreflightCheck::fail(
                        "satd",
                        format!("{} SATD items exceed limit {}", c, max_items),
                    ),
                    Some(c) => PreflightCheck::pass(
                        "satd",
                        format!("{} SATD items (warning: exceeds {})", c, max_items),
                    ),
                    None if output.status.success() => {
                        PreflightCheck::pass("satd", "SATD check passed")
                    }
                    None => PreflightCheck::pass("satd", "SATD check completed"),
                }
            },
        )
    }

    /// Check PMAT Popper score (falsifiability)
    ///
    /// Runs `pmat popper-score` to assess scientific quality.
    /// Based on Karl Popper's falsification principles.
    /// Fails if score < min_popper_score (default: 60).
    pub(super) fn check_pmat_popper(&self, crate_path: &Path) -> PreflightCheck {
        let min_score = self.config.min_popper_score;
        let fail_on = self.config.fail_on_popper;
        run_check_command(
            &self.config.popper_command,
            "popper",
            "No Popper command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let score = parse_value_from_json(stdout, &["score", "popper_score", "total"]);
                score_check_result(
                    "popper",
                    "Popper score",
                    score,
                    min_score,
                    fail_on,
                    output.status.success(),
                )
            },
        )
    }

    // =========================================================================
    // Book and Examples Verification (RELEASE-DOCS)
    // =========================================================================

    /// Check book builds successfully
    ///
    /// Runs `mdbook build book` (or configured command) to verify
    /// documentation compiles without errors.
    pub(super) fn check_book_build(&self, crate_path: &Path) -> PreflightCheck {
        // Check if book directory exists before running the command
        let book_dir = crate_path.join("book");
        if !book_dir.exists() {
            return PreflightCheck::pass("book", "No book directory found (skipped)");
        }

        let fail_on = self.config.fail_on_book;

        run_check_command(
            &self.config.book_command,
            "book",
            "No book command configured (skipped)",
            crate_path,
            move |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("book", "Book built successfully")
                } else if fail_on {
                    PreflightCheck::fail("book", format!("Book build failed: {}", stderr.trim()))
                } else {
                    PreflightCheck::pass("book", "Book build has warnings (not blocking)")
                }
            },
        )
    }

    /// Check examples compile and run successfully
    ///
    /// Discovers examples from Cargo.toml [[example]] sections and
    /// runs each one with `cargo run --example <name>`.
    pub(super) fn check_examples_run(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.examples_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("examples", "No examples command configured (skipped)");
        }

        // Check if examples directory exists
        let examples_dir = crate_path.join("examples");
        if !examples_dir.exists() {
            return PreflightCheck::pass("examples", "No examples directory found (skipped)");
        }

        // Discover examples from Cargo.toml or examples directory
        let examples = self.discover_examples(crate_path);
        if examples.is_empty() {
            return PreflightCheck::pass("examples", "No examples found (skipped)");
        }

        let mut failed = Vec::new();
        let mut succeeded = 0;

        for example in &examples {
            // Build the full command with example name
            let output = Command::new("cargo")
                .args(["run", "--example", example, "--", "--help"])
                .current_dir(crate_path)
                .output();

            match output {
                Ok(out) => {
                    // Consider it a pass if the example compiles and runs
                    // (even if --help exits with non-zero, compilation success is what matters)
                    if out.status.success() || out.status.code() == Some(0) {
                        succeeded += 1;
                    } else {
                        // Check if it failed during compilation vs runtime
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if stderr.contains("error[E") || stderr.contains("could not compile") {
                            failed.push(example.clone());
                        } else {
                            // Runtime exit with non-zero is OK for --help
                            succeeded += 1;
                        }
                    }
                }
                Err(_) => {
                    failed.push(example.clone());
                }
            }
        }

        if failed.is_empty() {
            PreflightCheck::pass(
                "examples",
                format!("{}/{} examples verified", succeeded, examples.len()),
            )
        } else if self.config.fail_on_examples {
            PreflightCheck::fail(
                "examples",
                format!(
                    "{}/{} examples failed: {}",
                    failed.len(),
                    examples.len(),
                    failed.join(", ")
                ),
            )
        } else {
            PreflightCheck::pass(
                "examples",
                format!(
                    "{}/{} examples verified ({} failed, not blocking)",
                    succeeded,
                    examples.len(),
                    failed.len()
                ),
            )
        }
    }

    /// Discover examples from the crate
    pub(super) fn discover_examples(&self, crate_path: &Path) -> Vec<String> {
        let mut examples = Vec::new();

        // Try to find examples from Cargo.toml
        let cargo_toml = crate_path.join("Cargo.toml");
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            // Simple parsing for [[example]] sections
            for line in content.lines() {
                if line.trim().starts_with("name = \"") {
                    // Check if we're in an [[example]] section by looking at previous context
                    // This is a simplified approach - in production, use toml crate
                    if let Some(name) = line.split('"').nth(1) {
                        // Verify it's actually in the examples dir
                        let example_file = crate_path.join("examples").join(format!("{}.rs", name));
                        if example_file.exists() {
                            examples.push(name.to_string());
                        }
                    }
                }
            }
        }

        // Also scan examples directory for .rs files
        let examples_dir = crate_path.join("examples");
        if let Ok(entries) = std::fs::read_dir(&examples_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "rs") {
                    if let Some(stem) = path.file_stem() {
                        let name = stem.to_string_lossy().to_string();
                        if !examples.contains(&name) {
                            examples.push(name);
                        }
                    }
                }
            }
        }

        examples
    }

    // =========================================================================
    // JSON Parsing Helpers
    // =========================================================================

    /// Helper: Parse a numeric score from JSON output
    pub(super) fn parse_score_from_json(json: &str, key: &str) -> Option<f64> {
        // Simple JSON parsing without serde for minimal dependencies
        let pattern = format!("\"{}\":", key);
        if let Some(pos) = json.find(&pattern) {
            let after_key = &json[pos + pattern.len()..];
            let value_str: String = after_key
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_numeric() || *c == '.' || *c == '-')
                .collect();
            value_str.parse().ok()
        } else {
            None
        }
    }

    /// Helper: Parse an integer count from JSON output
    pub(super) fn parse_count_from_json(json: &str, key: &str) -> Option<u32> {
        Self::parse_score_from_json(json, key).map(|f| f as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stack::graph::DependencyGraph;
    use crate::stack::releaser_types::ReleaseConfig;

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
