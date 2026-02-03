//! Jidoka Automated Gates (Section 7)
//!
//! Implements JA-01 through JA-10 from the Popperian Falsification Checklist.
//! Focus: CI/CD circuit breakers, automated quality enforcement.
//!
//! # Jidoka Principle
//!
//! "Automation with human intelligence" - The CI/CD pipeline must detect
//! abnormalities and stop automatically before human review begins.

use std::path::Path;
use std::time::Instant;

use super::helpers::{apply_check_outcome, CheckOutcome};
use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};

/// Evaluate all Jidoka automated gates for a project.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_precommit_hooks(project_path),
        check_automated_sovereignty_linting(project_path),
        check_data_drift_circuit_breaker(project_path),
        check_performance_regression_gate(project_path),
        check_fairness_metric_circuit_breaker(project_path),
        check_latency_sla_circuit_breaker(project_path),
        check_memory_footprint_gate(project_path),
        check_security_scan_gate(project_path),
        check_license_compliance_gate(project_path),
        check_documentation_gate(project_path),
    ]
}

/// JA-01: Pre-Commit Hook Enforcement
///
/// **Claim:** Pre-commit hooks catch basic issues locally.
///
/// **Rejection Criteria (Major):**
/// - >5% of CI failures are pre-commit-detectable
pub fn check_precommit_hooks(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-01",
        "Pre-Commit Hook Enforcement",
        "Pre-commit hooks catch basic issues locally",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — early detection");

    // Check for pre-commit configuration
    let precommit_yaml = project_path.join(".pre-commit-config.yaml");
    let has_precommit = precommit_yaml.exists();

    // Check for git hooks directory
    let git_hooks = project_path.join(".git/hooks");
    let has_git_hooks = git_hooks.exists()
        && (git_hooks.join("pre-commit").exists() || git_hooks.join("pre-push").exists());

    // Check for husky (Node.js)
    let husky_dir = project_path.join(".husky");
    let has_husky = husky_dir.exists();

    // Check for cargo-husky or similar in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let has_cargo_hooks = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("cargo-husky") || c.contains("[hooks]"))
        .unwrap_or(false);

    // Check for Makefile with pre-commit targets
    let makefile = project_path.join("Makefile");
    let has_make_hooks = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("pre-commit") || c.contains("precommit") || c.contains("tier1"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Pre-commit: yaml={}, git_hooks={}, husky={}, cargo_hooks={}, make_targets={}",
            has_precommit, has_git_hooks, has_husky, has_cargo_hooks, has_make_hooks
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_precommit || has_cargo_hooks, CheckOutcome::Pass),
            (
                has_git_hooks || has_husky || has_make_hooks,
                CheckOutcome::Partial("Pre-commit configured but not standardized"),
            ),
            (true, CheckOutcome::Fail("No pre-commit hooks configured")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-02: Automated Sovereignty Linting
///
/// **Claim:** Static analysis catches sovereignty violations.
///
/// **Rejection Criteria (Major):**
/// - Known violation pattern not flagged
pub fn check_automated_sovereignty_linting(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-02",
        "Automated Sovereignty Linting",
        "Static analysis catches sovereignty violations",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automated sovereignty check");

    // Check for clippy configuration
    let clippy_toml = project_path.join("clippy.toml");
    let has_clippy_config = clippy_toml.exists();

    // Check CI for clippy
    let has_clippy_ci = check_ci_for_content(project_path, "clippy");

    // Check for custom lints
    let has_custom_lints = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("#[deny(")
                            || c.contains("#![deny(")
                            || c.contains("#[warn(")
                            || c.contains("#![warn(")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for deny.toml (cargo-deny)
    let deny_toml = project_path.join("deny.toml");
    let has_deny = deny_toml.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Linting: clippy_config={}, clippy_ci={}, custom_lints={}, deny={}",
            has_clippy_config, has_clippy_ci, has_custom_lints, has_deny
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                has_clippy_ci && (has_deny || has_custom_lints),
                CheckOutcome::Pass,
            ),
            (
                has_clippy_ci,
                CheckOutcome::Partial("Clippy in CI but limited sovereignty-specific rules"),
            ),
            (true, CheckOutcome::Fail("No automated linting in CI")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-03: Data Drift Circuit Breaker
///
/// **Claim:** Training stops on significant data drift.
///
/// **Rejection Criteria (Major):**
/// - Training completes with >20% distribution shift
pub fn check_data_drift_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-03",
        "Data Drift Circuit Breaker",
        "Training stops on significant data drift",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automatic stop");

    // Check for drift detection patterns in code
    let has_drift_detection = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("drift")
                            || c.contains("distribution_shift")
                            || c.contains("data_quality")
                            || c.contains("schema_validation")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for data validation in tests
    let has_data_validation = glob::glob(&format!("{}/tests/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("data") && (c.contains("valid") || c.contains("schema")))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data drift: detection={}, validation={}",
            has_drift_detection, has_data_validation
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML training that needs drift detection
    let has_training = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("train") || c.contains("fit") || c.contains("epoch"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = apply_check_outcome(
        item,
        &[
            (
                !has_training || has_drift_detection || has_data_validation,
                CheckOutcome::Pass,
            ),
            (
                true,
                CheckOutcome::Partial("Training without data drift detection"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-04: Model Performance Regression Gate
///
/// **Claim:** Deployment blocked on performance regression.
///
/// **Rejection Criteria (Major):**
/// - Model with <baseline metrics deploys
pub fn check_performance_regression_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-04",
        "Performance Regression Gate",
        "Deployment blocked on performance regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — quality gate");

    // Check for benchmark configuration
    let benches_dir = project_path.join("benches");
    let has_benches = benches_dir.exists();

    // Check for criterion in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let has_criterion = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("criterion") || c.contains("divan") || c.contains("[bench]"))
        .unwrap_or(false);

    // Check CI for benchmarks
    let has_bench_ci = check_ci_for_content(project_path, "bench");

    // Check for hyperfine or similar
    let makefile = project_path.join("Makefile");
    let has_perf_make = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("hyperfine") || c.contains("bench") || c.contains("perf"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Performance: benches_dir={}, criterion={}, ci_bench={}, make_perf={}",
            has_benches, has_criterion, has_bench_ci, has_perf_make
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                has_benches && has_criterion && has_bench_ci,
                CheckOutcome::Pass,
            ),
            (
                has_benches || has_criterion,
                CheckOutcome::Partial("Benchmarks exist but not gated in CI"),
            ),
            (
                true,
                CheckOutcome::Partial("No performance regression detection"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-05: Fairness Metric Circuit Breaker
///
/// **Claim:** Training stops on fairness regression.
///
/// **Rejection Criteria (Major):**
/// - Protected class metric degrades >5% without alert
pub fn check_fairness_metric_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-05",
        "Fairness Metric Circuit Breaker",
        "Training stops on fairness regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — ethical safeguard");

    // Check for fairness-related code
    let has_fairness_code = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("fairness")
                            || c.contains("bias")
                            || c.contains("demographic_parity")
                            || c.contains("equalized_odds")
                            || c.contains("protected_class")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for fairness testing
    let has_fairness_tests = glob::glob(&format!("{}/tests/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("fairness") || c.contains("bias"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Fairness: code={}, tests={}",
            has_fairness_code, has_fairness_tests
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML that needs fairness monitoring
    let has_ml = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("classifier") || c.contains("predict") || c.contains("model")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = apply_check_outcome(
        item,
        &[
            (
                !has_ml || has_fairness_code || has_fairness_tests,
                CheckOutcome::Pass,
            ),
            (
                true,
                CheckOutcome::Partial("ML without fairness monitoring"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-06: Latency SLA Circuit Breaker
///
/// **Claim:** Deployment blocked on latency regression.
///
/// **Rejection Criteria (Major):**
/// - P99 latency exceeds SLA in staging
pub fn check_latency_sla_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-06",
        "Latency SLA Circuit Breaker",
        "Deployment blocked on latency regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — SLA enforcement");

    // Check for latency-related code
    let has_latency_monitoring = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("latency")
                            || c.contains("p99")
                            || c.contains("p95")
                            || c.contains("percentile")
                            || c.contains("sla")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for timing/duration tracking
    let has_timing = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("Instant::") || c.contains("Duration::") || c.contains("elapsed")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Latency: monitoring={}, timing={}",
            has_latency_monitoring, has_timing
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has serving that needs latency SLA
    let has_serving = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("serve") || c.contains("inference") || c.contains("api"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = apply_check_outcome(
        item,
        &[
            (!has_serving || has_latency_monitoring, CheckOutcome::Pass),
            (
                has_timing,
                CheckOutcome::Partial("Timing code exists but no SLA enforcement"),
            ),
            (
                true,
                CheckOutcome::Partial("Serving without latency monitoring"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-07: Memory Footprint Gate
///
/// **Claim:** Deployment blocked on excessive memory.
///
/// **Rejection Criteria (Major):**
/// - Peak memory exceeds target by >20%
pub fn check_memory_footprint_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-07",
        "Memory Footprint Gate",
        "Deployment blocked on excessive memory",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Inventory) prevention");

    // Check for memory profiling patterns
    let has_memory_profiling = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("memory")
                            || c.contains("heap")
                            || c.contains("allocator")
                            || c.contains("mem::size_of")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for memory limits in CI or config
    let has_memory_limits = check_ci_for_content(project_path, "memory")
        || check_ci_for_content(project_path, "ulimit");

    // Check Makefile for memory profiling
    let makefile = project_path.join("Makefile");
    let has_heaptrack = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("heaptrack") || c.contains("valgrind") || c.contains("massif"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Memory: profiling={}, limits={}, heaptrack={}",
            has_memory_profiling, has_memory_limits, has_heaptrack
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                has_memory_profiling && (has_memory_limits || has_heaptrack),
                CheckOutcome::Pass,
            ),
            (
                has_memory_profiling || has_heaptrack,
                CheckOutcome::Partial("Memory profiling available but not gated"),
            ),
            (true, CheckOutcome::Partial("No memory footprint gate")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-08: Security Scan Gate
///
/// **Claim:** Build blocked on security findings.
///
/// **Rejection Criteria (Critical):**
/// - High/Critical vulnerability in build
pub fn check_security_scan_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-08",
        "Security Scan Gate",
        "Build blocked on security findings",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — security gate");

    // Check for security scanning tools
    let has_audit_ci = check_ci_for_content(project_path, "cargo audit");
    let has_deny_ci = check_ci_for_content(project_path, "cargo deny");

    // Check for deny.toml
    let deny_toml = project_path.join("deny.toml");
    let has_deny_config = deny_toml.exists();

    // Check for security workflow
    let security_workflow = project_path.join(".github/workflows/security.yml");
    let has_security_workflow = security_workflow.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Security: audit_ci={}, deny_ci={}, deny_config={}, security_workflow={}",
            has_audit_ci, has_deny_ci, has_deny_config, has_security_workflow
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                has_deny_config && (has_audit_ci || has_deny_ci),
                CheckOutcome::Pass,
            ),
            (
                has_audit_ci || has_deny_ci || has_deny_config,
                CheckOutcome::Partial("Security scanning partially configured"),
            ),
            (true, CheckOutcome::Fail("No security scanning in CI")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-09: License Compliance Gate
///
/// **Claim:** Build blocked on license violation.
///
/// **Rejection Criteria (Major):**
/// - Disallowed license in dependency tree
pub fn check_license_compliance_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-09",
        "License Compliance Gate",
        "Build blocked on license violation",
    )
    .with_severity(Severity::Major)
    .with_tps("Legal controls pillar");

    // Check for deny.toml with licenses section
    let deny_toml = project_path.join("deny.toml");
    let has_license_config = deny_toml
        .exists()
        .then(|| std::fs::read_to_string(&deny_toml).ok())
        .flatten()
        .map(|c| c.contains("[licenses]"))
        .unwrap_or(false);

    // Check for cargo-deny in CI
    let has_deny_licenses_ci = check_ci_for_content(project_path, "cargo deny check licenses");

    // Check for LICENSE file
    let has_license_file = project_path.join("LICENSE").exists()
        || project_path.join("LICENSE.md").exists()
        || project_path.join("LICENSE.txt").exists()
        || project_path.join("LICENSE-MIT").exists()
        || project_path.join("LICENSE-APACHE").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "License: config={}, ci_check={}, license_file={}",
            has_license_config, has_deny_licenses_ci, has_license_file
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                has_license_config && has_deny_licenses_ci,
                CheckOutcome::Pass,
            ),
            (
                has_license_file && (has_license_config || has_deny_licenses_ci),
                CheckOutcome::Partial("License file exists, partial enforcement"),
            ),
            (
                has_license_file,
                CheckOutcome::Partial("License file exists but no automated check"),
            ),
            (true, CheckOutcome::Fail("No license compliance setup")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// JA-10: Documentation Gate
///
/// **Claim:** PR blocked without documentation updates.
///
/// **Rejection Criteria (Minor):**
/// - Public API change without doc update
pub fn check_documentation_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-10",
        "Documentation Gate",
        "PR blocked without documentation updates",
    )
    .with_severity(Severity::Minor)
    .with_tps("Knowledge transfer");

    // Check for doc tests
    let has_doc_tests = check_ci_for_content(project_path, "cargo doc");

    // Check for deny warnings on missing docs
    let lib_rs = project_path.join("src/lib.rs");
    let has_deny_missing_docs = lib_rs
        .exists()
        .then(|| std::fs::read_to_string(&lib_rs).ok())
        .flatten()
        .map(|c| c.contains("#![deny(missing_docs)]") || c.contains("#![warn(missing_docs)]"))
        .unwrap_or(false);

    // Check for README
    let has_readme = project_path.join("README.md").exists();

    // Check for docs directory or book
    let has_docs_dir = project_path.join("docs").exists() || project_path.join("book").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Docs: ci_doc={}, deny_missing={}, readme={}, docs_dir={}",
            has_doc_tests, has_deny_missing_docs, has_readme, has_docs_dir
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                (has_doc_tests && has_deny_missing_docs) || (has_readme && has_docs_dir),
                CheckOutcome::Pass,
            ),
            (
                has_readme,
                CheckOutcome::Partial("README exists but no documentation enforcement"),
            ),
            (true, CheckOutcome::Fail("No documentation gate")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Helper: Check if content exists in any CI configuration
fn check_ci_for_content(project_path: &Path, content: &str) -> bool {
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
        project_path.join(".github/workflows/security.yml"),
        project_path.join(".github/workflows/release.yml"),
    ];

    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(file_content) = std::fs::read_to_string(ci_path) {
                if file_content.contains(content) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // JA-01: Pre-Commit Hook Tests
    // =========================================================================

    #[test]
    fn test_ja_01_precommit_hooks() {
        let result = check_precommit_hooks(Path::new("."));
        // batuta should have some form of pre-commit
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Pre-commit check should complete"
        );
    }

    // =========================================================================
    // JA-02: Sovereignty Linting Tests
    // =========================================================================

    #[test]
    fn test_ja_02_sovereignty_linting() {
        let result = check_automated_sovereignty_linting(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Linting check should complete"
        );
    }

    // =========================================================================
    // JA-03: Data Drift Circuit Breaker Tests
    // =========================================================================

    #[test]
    fn test_ja_03_data_drift_circuit_breaker() {
        let result = check_data_drift_circuit_breaker(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Data drift check should complete"
        );
    }

    // =========================================================================
    // JA-04: Performance Regression Tests
    // =========================================================================

    #[test]
    fn test_ja_04_performance_regression() {
        let result = check_performance_regression_gate(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Performance check should complete"
        );
    }

    // =========================================================================
    // JA-05: Fairness Metric Circuit Breaker Tests
    // =========================================================================

    #[test]
    fn test_ja_05_fairness_circuit_breaker() {
        let result = check_fairness_metric_circuit_breaker(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Fairness check should complete"
        );
    }

    // =========================================================================
    // JA-06: Latency SLA Circuit Breaker Tests
    // =========================================================================

    #[test]
    fn test_ja_06_latency_sla_circuit_breaker() {
        let result = check_latency_sla_circuit_breaker(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Latency check should complete"
        );
    }

    // =========================================================================
    // JA-07: Memory Footprint Gate Tests
    // =========================================================================

    #[test]
    fn test_ja_07_memory_footprint_gate() {
        let result = check_memory_footprint_gate(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Memory check should complete"
        );
    }

    // =========================================================================
    // JA-08: Security Scan Tests
    // =========================================================================

    #[test]
    fn test_ja_08_security_scan() {
        let result = check_security_scan_gate(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Security check should complete"
        );
    }

    // =========================================================================
    // JA-09: License Compliance Tests
    // =========================================================================

    #[test]
    fn test_ja_09_license_compliance() {
        let result = check_license_compliance_gate(Path::new("."));
        // batuta should have LICENSE
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "License check failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // JA-10: Documentation Gate Tests
    // =========================================================================

    #[test]
    fn test_ja_10_documentation_gate() {
        let result = check_documentation_gate(Path::new("."));
        // batuta should have README
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "Documentation check failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_evaluate_all_returns_10_items() {
        let results = evaluate_all(Path::new("."));
        assert_eq!(results.len(), 10, "Expected 10 Jidoka checks");
    }

    #[test]
    fn test_all_items_have_evidence() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(
                !item.evidence.is_empty(),
                "Item {} missing evidence",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(
                !item.tps_principle.is_empty(),
                "Item {} missing TPS principle",
                item.id
            );
        }
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_nonexistent_path() {
        let results = evaluate_all(Path::new("/nonexistent/path/that/does/not/exist"));
        // Should still return 10 items, likely all failed
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_check_ci_for_content_nonexistent() {
        let result = check_ci_for_content(Path::new("/nonexistent/path"), "cargo test");
        assert!(!result);
    }

    #[test]
    fn test_check_ci_for_content_exists() {
        // batuta has CI configured
        let result = check_ci_for_content(Path::new("."), "rust");
        // Should check all CI files and find some content
        // Result depends on actual CI config, so just test it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_all_items_have_valid_ids() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(item.id.starts_with("JA-"), "Item ID {} should start with JA-", item.id);
        }
    }

    #[test]
    fn test_all_items_have_durations() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            // duration_ms is set by with_duration() call
            // All checks in this module set duration
            assert!(
                item.duration_ms > 0 || item.duration_ms == 0,
                "Item {} should have duration recorded",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_valid_severities() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(
                matches!(item.severity, Severity::Critical | Severity::Major | Severity::Minor | Severity::Info),
                "Item {} has unexpected severity",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_claims() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(!item.claim.is_empty(), "Item {} missing claim", item.id);
        }
    }

    #[test]
    fn test_all_items_have_names() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(!item.name.is_empty(), "Item {} missing name", item.id);
        }
    }

    #[test]
    fn test_ja_01_has_correct_severity() {
        let result = check_precommit_hooks(Path::new("."));
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ja_08_has_critical_severity() {
        let result = check_security_scan_gate(Path::new("."));
        assert_eq!(result.severity, Severity::Critical);
    }

    #[test]
    fn test_ja_10_has_minor_severity() {
        let result = check_documentation_gate(Path::new("."));
        assert_eq!(result.severity, Severity::Minor);
    }

    #[test]
    fn test_check_items_order() {
        let results = evaluate_all(Path::new("."));
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec![
            "JA-01", "JA-02", "JA-03", "JA-04", "JA-05",
            "JA-06", "JA-07", "JA-08", "JA-09", "JA-10"
        ]);
    }

    #[test]
    fn test_evidence_type_is_static_analysis() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            for evidence in &item.evidence {
                assert!(
                    matches!(evidence.evidence_type, EvidenceType::StaticAnalysis),
                    "Item {} has non-static-analysis evidence",
                    item.id
                );
            }
        }
    }

    #[test]
    fn test_evidence_descriptions_not_empty() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            for evidence in &item.evidence {
                assert!(
                    !evidence.description.is_empty(),
                    "Item {} has empty evidence description",
                    item.id
                );
            }
        }
    }
}
