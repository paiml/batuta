//! Section 3: Hypothesis-Driven & Equation-Driven Development (HDD-01 to EDD-03)
//!
//! Implements Scientific Method integration for ML development.
//!
//! # TPS Principles
//!
//! - **Scientific Method**: Falsifiable hypotheses, pre-registration
//! - **Muda**: Overprocessing prevention via baseline comparison
//! - **Kaizen**: Learning from negative results

use super::helpers::{apply_check_outcome, CheckOutcome};
use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Hypothesis-Driven Development checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_hypothesis_statement(project_path),
        check_baseline_comparison(project_path),
        check_gold_reproducibility(project_path),
        check_random_seed_documentation(project_path),
        check_environment_containerization(project_path),
        check_data_version_control(project_path),
        check_statistical_significance(project_path),
        check_ablation_study(project_path),
        check_negative_result_documentation(project_path),
        check_metric_preregistration(project_path),
        check_equation_verification(project_path),
        check_emc_completeness(project_path),
        check_numerical_analytical_validation(project_path),
    ]
}

/// HDD-01: Hypothesis Statement Requirement
///
/// **Claim:** Every model change PR includes falsifiable hypothesis.
///
/// **Rejection Criteria (Major):**
/// - Model PR without "Hypothesis:" section
pub fn check_hypothesis_statement(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-01",
        "Hypothesis Statement Requirement",
        "Model change PRs include falsifiable hypothesis",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific Method integration");

    // Check for PR template with hypothesis section
    let pr_templates = [
        project_path.join(".github/PULL_REQUEST_TEMPLATE.md"),
        project_path.join(".github/pull_request_template.md"),
        project_path.join("docs/PR_TEMPLATE.md"),
    ];

    let has_pr_template = pr_templates.iter().any(|p| p.exists());
    let has_hypothesis_section = pr_templates.iter().filter(|p| p.exists()).any(|p| {
        std::fs::read_to_string(p)
            .ok()
            .map(|c| {
                c.to_lowercase().contains("hypothesis")
                    || c.contains("## Hypothesis")
                    || c.contains("### Hypothesis")
            })
            .unwrap_or(false)
    });

    // Check for hypothesis documentation in codebase
    let has_hypothesis_docs = check_for_pattern(
        project_path,
        &[
            "hypothesis:",
            "Hypothesis:",
            "H0:",
            "H1:",
            "null_hypothesis",
        ],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Hypothesis: pr_template={}, hypothesis_section={}, docs={}",
            has_pr_template, has_hypothesis_section, has_hypothesis_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml_project = check_for_pattern(project_path, &["model", "train", "predict"]);
    item = apply_check_outcome(item, &[
        (has_hypothesis_section, CheckOutcome::Pass),
        (has_pr_template || has_hypothesis_docs, CheckOutcome::Partial("PR template exists but missing hypothesis section")),
        (!is_ml_project, CheckOutcome::Pass),
        (true, CheckOutcome::Partial("No hypothesis requirement in PR workflow")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-02: Baseline Comparison Requirement
///
/// **Claim:** Complex models must beat simple baselines to be merged.
///
/// **Rejection Criteria (Major):**
/// - Transformer without RF baseline, <5% improvement without justification
pub fn check_baseline_comparison(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-02",
        "Baseline Comparison Requirement",
        "Complex models beat simple baselines",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Overprocessing) prevention");

    // Check for baseline comparison in tests or benchmarks
    let has_baseline_tests = check_for_pattern(
        project_path,
        &[
            "baseline",
            "Baseline",
            "simple_model",
            "compare_to_baseline",
        ],
    );

    // Check for benchmark infrastructure
    let has_benchmarks = project_path.join("benches").exists()
        || check_for_pattern(project_path, &["#[bench]", "criterion", "benchmark"]);

    // Check for model comparison documentation
    let has_comparison_docs = check_for_pattern(
        project_path,
        &["vs_baseline", "improvement_over", "comparison"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Baseline: tests={}, benchmarks={}, comparison_docs={}",
            has_baseline_tests, has_benchmarks, has_comparison_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["neural", "transformer", "deep_learning"]);
    item = apply_check_outcome(item, &[
        (has_baseline_tests && has_benchmarks, CheckOutcome::Pass),
        (has_baseline_tests || has_comparison_docs, CheckOutcome::Partial("Some baseline comparison infrastructure")),
        (!is_ml, CheckOutcome::Pass),
        (true, CheckOutcome::Partial("Complex models without baseline comparison")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-03: Gold Standard Reproducibility
///
/// **Claim:** `make reproduce` recreates training results from scratch.
///
/// **Rejection Criteria (Critical):**
/// - Build fails from clean state, metrics differ >1%
pub fn check_gold_reproducibility(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-03",
        "Gold Standard Reproducibility",
        "make reproduce recreates training results",
    )
    .with_severity(Severity::Critical)
    .with_tps("Scientific reproducibility");

    // Check for reproduce target in Makefile
    let makefile = project_path.join("Makefile");
    let has_reproduce_target = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("reproduce:") || c.contains("reproduce :"))
        .unwrap_or(false);

    // Check for reproduction documentation
    let has_repro_docs = check_for_pattern(
        project_path,
        &["REPRODUCIBILITY", "reproduce", "replication"],
    );

    // Check for CI reproduction
    let has_ci_repro = check_ci_for_pattern(project_path, &["reproduce", "replication"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Reproducibility: make_target={}, docs={}, ci={}",
            has_reproduce_target, has_repro_docs, has_ci_repro
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["train", "model", "weights"]);
    item = apply_check_outcome(item, &[
        (has_reproduce_target && has_ci_repro, CheckOutcome::Pass),
        (has_reproduce_target || has_repro_docs, CheckOutcome::Partial("Reproduction target exists (not in CI)")),
        (!is_ml, CheckOutcome::Pass),
        (true, CheckOutcome::Partial("No reproduction infrastructure")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-04: Random Seed Documentation
///
/// **Claim:** All stochastic operations have documented, pinned seeds.
///
/// **Rejection Criteria (Major):**
/// - Any stochastic operation without explicit seed
pub fn check_random_seed_documentation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-04",
        "Random Seed Documentation",
        "Stochastic operations have pinned seeds",
    )
    .with_severity(Severity::Major)
    .with_tps("Deterministic reproducibility");

    // Check for seed pinning in code
    let has_seed_pinning = check_for_pattern(
        project_path,
        &["seed", "Seed", "RANDOM_SEED", "rng_seed", "set_seed"],
    );

    // Check for StdRng usage (seeded RNG)
    let has_seeded_rng = check_for_pattern(
        project_path,
        &["StdRng", "SeedableRng", "from_seed", "seed_from_u64"],
    );

    // Check for seed documentation
    let has_seed_docs = check_for_pattern(
        project_path,
        &["seed=", "SEED:", "random seed", "deterministic"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Seeds: pinning={}, seeded_rng={}, docs={}",
            has_seed_pinning, has_seeded_rng, has_seed_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let uses_random = check_for_pattern(project_path, &["rand::", "random", "Rng", "thread_rng"]);
    item = apply_check_outcome(item, &[
        (!uses_random, CheckOutcome::Pass),
        (has_seeded_rng && has_seed_pinning, CheckOutcome::Pass),
        (has_seed_pinning, CheckOutcome::Partial("Seed usage found (verify documentation)")),
        (true, CheckOutcome::Partial("Random usage without explicit seed pinning")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-05: Environment Containerization
///
/// **Claim:** Training environment is fully containerized and versioned.
///
/// **Rejection Criteria (Major):**
/// - Dockerfile missing, unpinned dependencies
pub fn check_environment_containerization(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-05",
        "Environment Containerization",
        "Environment fully containerized and versioned",
    )
    .with_severity(Severity::Major)
    .with_tps("Silver → Gold reproducibility");

    // Check for Dockerfile
    let has_dockerfile =
        project_path.join("Dockerfile").exists() || project_path.join("docker/Dockerfile").exists();

    // Check for docker-compose
    let has_compose = project_path.join("docker-compose.yml").exists()
        || project_path.join("docker-compose.yaml").exists();

    // Check for Cargo.lock (Rust dependency pinning)
    let has_lock_file = project_path.join("Cargo.lock").exists();

    // Check for nix/devenv
    let has_nix = project_path.join("flake.nix").exists()
        || project_path.join("shell.nix").exists()
        || project_path.join("devenv.nix").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Container: dockerfile={}, compose={}, lock={}, nix={}",
            has_dockerfile, has_compose, has_lock_file, has_nix
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(item, &[
        ((has_dockerfile || has_nix) && has_lock_file, CheckOutcome::Pass),
        (has_lock_file, CheckOutcome::Partial("Dependencies locked but no containerization")),
        (true, CheckOutcome::Partial("No environment containerization")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-06: Data Version Control
///
/// **Claim:** Training data is versioned with content-addressable storage.
///
/// **Rejection Criteria (Major):**
/// - Any data modification not captured in version control
pub fn check_data_version_control(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-06",
        "Data Version Control",
        "Training data versioned with content-addressable storage",
    )
    .with_severity(Severity::Major)
    .with_tps("Reproducibility requirement");

    // Check for DVC
    let has_dvc = project_path.join(".dvc").exists() || project_path.join("dvc.yaml").exists();

    // Check for data versioning patterns
    let has_data_versioning = check_for_pattern(
        project_path,
        &["data_version", "dataset_hash", "content_hash"],
    );

    // Check for data directory with hashes
    let data_dir = project_path.join("data");
    let has_data_dir = data_dir.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data versioning: dvc={}, versioning_code={}, data_dir={}",
            has_dvc, has_data_versioning, has_data_dir
        ),
        data: None,
        files: Vec::new(),
    });

    let uses_data = check_for_pattern(project_path, &["load_data", "Dataset", "DataLoader"]);
    item = apply_check_outcome(item, &[
        (!uses_data, CheckOutcome::Pass),
        (has_dvc, CheckOutcome::Pass),
        (has_data_versioning, CheckOutcome::Partial("Data versioning patterns found (no DVC)")),
        (true, CheckOutcome::Partial("Data handling without version control")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-07: Statistical Significance Requirement
///
/// **Claim:** Performance claims include statistical significance tests.
///
/// **Rejection Criteria (Major):**
/// - p ≥ 0.05, no confidence interval
pub fn check_statistical_significance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-07",
        "Statistical Significance Requirement",
        "Performance claims include significance tests",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific rigor");

    // Check for statistical testing
    let has_stats = check_for_pattern(
        project_path,
        &[
            "p_value",
            "p-value",
            "confidence_interval",
            "t_test",
            "significance",
        ],
    );

    // Check for effect size
    let has_effect_size = check_for_pattern(
        project_path,
        &["effect_size", "cohen_d", "glass_delta", "hedges_g"],
    );

    // Check for statistical library
    let has_stats_lib =
        check_for_pattern(project_path, &["statrs", "statistical", "hypothesis_test"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Statistics: testing={}, effect_size={}, lib={}",
            has_stats, has_effect_size, has_stats_lib
        ),
        data: None,
        files: Vec::new(),
    });

    let has_perf_claims =
        check_for_pattern(project_path, &["accuracy", "F1", "precision", "recall"]);
    item = apply_check_outcome(item, &[
        (!has_perf_claims, CheckOutcome::Pass),
        (has_stats && has_effect_size, CheckOutcome::Pass),
        (has_stats, CheckOutcome::Partial("Statistical testing (missing effect size)")),
        (true, CheckOutcome::Partial("Performance metrics without significance testing")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-08: Ablation Study Requirement
///
/// **Claim:** Multi-component changes include ablation studies.
///
/// **Rejection Criteria (Major):**
/// - >2 model changes without per-component analysis
pub fn check_ablation_study(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-08",
        "Ablation Study Requirement",
        "Multi-component changes include ablation studies",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific Method — isolation of variables");

    // Check for ablation documentation
    let has_ablation = check_for_pattern(
        project_path,
        &[
            "ablation",
            "Ablation",
            "component_analysis",
            "feature_importance",
        ],
    );

    // Check for sensitivity analysis
    let has_sensitivity = check_for_pattern(
        project_path,
        &["sensitivity", "hyperparameter_sweep", "grid_search"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Ablation: studies={}, sensitivity={}",
            has_ablation, has_sensitivity
        ),
        data: None,
        files: Vec::new(),
    });

    let has_complex_models = check_for_pattern(
        project_path,
        &["neural", "transformer", "ensemble", "multi_layer"],
    );
    item = apply_check_outcome(item, &[
        (!has_complex_models, CheckOutcome::Pass),
        (has_ablation, CheckOutcome::Pass),
        (has_sensitivity, CheckOutcome::Partial("Sensitivity analysis (no formal ablation)")),
        (true, CheckOutcome::Partial("Complex models without ablation studies")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-09: Negative Result Documentation
///
/// **Claim:** Failed experiments are documented, not just successes.
///
/// **Rejection Criteria (Minor):**
/// - Experiment log shows only successful attempts
pub fn check_negative_result_documentation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-09",
        "Negative Result Documentation",
        "Failed experiments documented",
    )
    .with_severity(Severity::Minor)
    .with_tps("Kaizen — learning from failures");

    // Check for experiment logs
    let has_experiment_log = project_path.join("experiments/").exists()
        || project_path.join("logs/experiments/").exists()
        || check_for_pattern(project_path, &["experiment_log", "run_history"]);

    // Check for negative result documentation
    let has_negative_docs = check_for_pattern(
        project_path,
        &[
            "failed_experiment",
            "negative_result",
            "did_not_work",
            "unsuccessful",
        ],
    );

    // Check for ADR (Architecture Decision Records)
    let has_adr =
        project_path.join("docs/adr/").exists() || project_path.join("docs/decisions/").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Negative results: log={}, docs={}, adr={}",
            has_experiment_log, has_negative_docs, has_adr
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(item, &[
        (has_negative_docs || has_adr, CheckOutcome::Pass),
        (has_experiment_log, CheckOutcome::Partial("Experiment logging (check for negative results)")),
        (true, CheckOutcome::Partial("No negative result documentation")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// HDD-10: Pre-registration of Metrics
///
/// **Claim:** Evaluation metrics defined before experimentation.
///
/// **Rejection Criteria (Minor):**
/// - Metric definition after experiment commit
pub fn check_metric_preregistration(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-10",
        "Pre-registration of Metrics",
        "Metrics defined before experimentation",
    )
    .with_severity(Severity::Minor)
    .with_tps("Scientific pre-registration");

    // Check for metrics definition in config
    let has_metric_config = check_for_pattern(
        project_path,
        &["metrics:", "evaluation_metrics", "target_metric"],
    );

    // Check for pre-registration documentation
    let has_prereg = check_for_pattern(
        project_path,
        &["pre_registration", "preregistration", "planned_metrics"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Pre-registration: config={}, docs={}",
            has_metric_config, has_prereg
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["accuracy", "loss", "evaluate"]);
    item = apply_check_outcome(item, &[
        (has_prereg, CheckOutcome::Pass),
        (has_metric_config, CheckOutcome::Partial("Metrics in config (verify pre-registration)")),
        (!is_ml, CheckOutcome::Pass),
        (true, CheckOutcome::Partial("No metric pre-registration")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// EDD-01: Equation Verification Before Implementation
///
/// **Claim:** Every simulation has analytically verified governing equation.
///
/// **Rejection Criteria (Major):**
/// - Simulation without EMC or analytical derivation
pub fn check_equation_verification(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-01",
        "Equation Verification Before Implementation",
        "Simulations have verified governing equations",
    )
    .with_severity(Severity::Major)
    .with_tps("EDD — prove first, implement second");

    // Check for EMC (Equation Model Cards)
    let has_emc = project_path.join("docs/emc/").exists()
        || check_for_pattern(
            project_path,
            &["equation_model_card", "EMC", "governing_equation"],
        );

    // Check for mathematical documentation
    let has_math_docs = check_for_pattern(
        project_path,
        &["derivation", "proof", "analytical_solution", "LaTeX"],
    );

    // Check for simulation code
    let has_simulation = check_for_pattern(
        project_path,
        &["simulate", "Simulator", "physics", "dynamics"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "EDD: emc={}, math_docs={}, simulation={}",
            has_emc, has_math_docs, has_simulation
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(item, &[
        (!has_simulation, CheckOutcome::Pass),
        (has_emc && has_math_docs, CheckOutcome::Pass),
        (has_emc || has_math_docs, CheckOutcome::Partial("Partial equation documentation")),
        (true, CheckOutcome::Partial("Simulation without equation verification")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// EDD-02: Equation Model Card (EMC) Completeness
///
/// **Claim:** Every shared simulation includes complete EMC.
///
/// **Rejection Criteria (Major):**
/// - EMC missing required sections
pub fn check_emc_completeness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-02",
        "Equation Model Card Completeness",
        "Simulations include complete EMC",
    )
    .with_severity(Severity::Major)
    .with_tps("Governance — equation traceability");

    // Check for EMC directory
    let emc_dir = project_path.join("docs/emc/");
    let has_emc_dir = emc_dir.exists();

    // Check EMC completeness (governing equations, validity, derivation, stability, tests)
    let required_sections = [
        "governing",
        "validity",
        "derivation",
        "stability",
        "verification",
    ];

    let mut sections_found = 0;
    if let Ok(entries) = glob::glob(&format!("{}/**/*.md", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for section in &required_sections {
                    if content.to_lowercase().contains(section) {
                        sections_found += 1;
                    }
                }
            }
        }
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "EMC: dir={}, sections_found={}/{}",
            has_emc_dir,
            sections_found,
            required_sections.len()
        ),
        data: None,
        files: Vec::new(),
    });

    let has_simulation = check_for_pattern(project_path, &["simulate", "Simulator"]);
    item = apply_check_outcome(item, &[
        (!has_simulation, CheckOutcome::Pass),
        (has_emc_dir && sections_found >= 4, CheckOutcome::Pass),
        (sections_found >= 2, CheckOutcome::Partial("Partial EMC documentation")),
        (true, CheckOutcome::Partial("Missing EMC documentation")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// EDD-03: Numerical vs Analytical Validation
///
/// **Claim:** Simulation results match analytical solutions within tolerance.
///
/// **Rejection Criteria (Major):**
/// - Numerical-analytical deviation >1e-6
pub fn check_numerical_analytical_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-03",
        "Numerical vs Analytical Validation",
        "Simulation matches analytical solutions",
    )
    .with_severity(Severity::Major)
    .with_tps("Verification — known solutions");

    // Check for analytical validation tests
    let has_analytical_tests = check_for_pattern(
        project_path,
        &[
            "analytical_solution",
            "exact_solution",
            "closed_form",
            "validate_against",
        ],
    );

    // Check for tolerance specification
    let has_tolerance = check_for_pattern(
        project_path,
        &["tolerance", "epsilon", "1e-6", "assert_relative_eq"],
    );

    // Check for verification test suite
    let has_verification = check_for_pattern(
        project_path,
        &[
            "verification_test",
            "numerical_validation",
            "convergence_test",
        ],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Validation: analytical={}, tolerance={}, verification={}",
            has_analytical_tests, has_tolerance, has_verification
        ),
        data: None,
        files: Vec::new(),
    });

    let has_simulation = check_for_pattern(project_path, &["simulate", "numerical"]);
    item = apply_check_outcome(item, &[
        (!has_simulation, CheckOutcome::Pass),
        (has_analytical_tests && has_tolerance, CheckOutcome::Pass),
        (has_verification || has_analytical_tests, CheckOutcome::Partial("Some validation (verify tolerance)")),
        (true, CheckOutcome::Partial("Numerical code without analytical validation")),
    ]);

    item.with_duration(start.elapsed().as_millis() as u64)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::files_contain_pattern(
        project_path,
        &["src/**/*.rs", "**/*.yaml", "**/*.toml", "**/*.json", "**/*.md"],
        patterns,
    )
}

fn check_ci_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::ci_contains_pattern(project_path, patterns)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_all_returns_13_items() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 13);
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        for item in items {
            assert!(
                !item.tps_principle.is_empty(),
                "Item {} missing TPS principle",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_evidence() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        for item in items {
            assert!(
                !item.evidence.is_empty(),
                "Item {} missing evidence",
                item.id
            );
        }
    }

    #[test]
    fn test_hdd_01_hypothesis() {
        let path = PathBuf::from(".");
        let item = check_hypothesis_statement(&path);
        assert_eq!(item.id, "HDD-01");
        assert_eq!(item.severity, Severity::Major);
    }

    #[test]
    fn test_hdd_03_reproducibility() {
        let path = PathBuf::from(".");
        let item = check_gold_reproducibility(&path);
        assert_eq!(item.id, "HDD-03");
        assert_eq!(item.severity, Severity::Critical);
    }

    #[test]
    fn test_edd_01_equation() {
        let path = PathBuf::from(".");
        let item = check_equation_verification(&path);
        assert_eq!(item.id, "EDD-01");
        assert_eq!(item.severity, Severity::Major);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_hdd_02_baseline() {
        let path = PathBuf::from(".");
        let item = check_baseline_comparison(&path);
        assert_eq!(item.id, "HDD-02");
    }

    #[test]
    fn test_hdd_04_random_seed() {
        let path = PathBuf::from(".");
        let item = check_random_seed_documentation(&path);
        assert_eq!(item.id, "HDD-04");
    }

    #[test]
    fn test_hdd_05_containerization() {
        let path = PathBuf::from(".");
        let item = check_environment_containerization(&path);
        assert_eq!(item.id, "HDD-05");
        assert_eq!(item.severity, Severity::Major);
    }

    #[test]
    fn test_hdd_06_data_version_control() {
        let path = PathBuf::from(".");
        let item = check_data_version_control(&path);
        assert_eq!(item.id, "HDD-06");
    }

    #[test]
    fn test_hdd_07_statistical_significance() {
        let path = PathBuf::from(".");
        let item = check_statistical_significance(&path);
        assert_eq!(item.id, "HDD-07");
    }

    #[test]
    fn test_hdd_08_ablation_study() {
        let path = PathBuf::from(".");
        let item = check_ablation_study(&path);
        assert_eq!(item.id, "HDD-08");
    }

    #[test]
    fn test_hdd_09_negative_results() {
        let path = PathBuf::from(".");
        let item = check_negative_result_documentation(&path);
        assert_eq!(item.id, "HDD-09");
    }

    #[test]
    fn test_hdd_10_metric_preregistration() {
        let path = PathBuf::from(".");
        let item = check_metric_preregistration(&path);
        assert_eq!(item.id, "HDD-10");
    }

    #[test]
    fn test_edd_02_emc_completeness() {
        let path = PathBuf::from(".");
        let item = check_emc_completeness(&path);
        assert_eq!(item.id, "EDD-02");
    }

    #[test]
    fn test_edd_03_numerical_analytical() {
        let path = PathBuf::from(".");
        let item = check_numerical_analytical_validation(&path);
        assert_eq!(item.id, "EDD-03");
    }

    #[test]
    fn test_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 13);
        // All should have a status (pass or fail)
        for item in &items {
            assert!(!item.id.is_empty());
        }
    }

    #[test]
    fn test_temp_dir_with_docker() {
        let temp_dir = std::env::temp_dir().join("test_docker_env");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create a Dockerfile
        std::fs::write(temp_dir.join("Dockerfile"), "FROM rust:1.70").unwrap();

        let item = check_environment_containerization(&temp_dir);
        assert_eq!(item.id, "HDD-05");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_dvc() {
        let temp_dir = std::env::temp_dir().join("test_dvc_env");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".dvc")).unwrap();

        std::fs::write(temp_dir.join(".dvc/config"), "[core]").unwrap();

        let item = check_data_version_control(&temp_dir);
        assert_eq!(item.id, "HDD-06");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_experiment_tracking() {
        let temp_dir = std::env::temp_dir().join("test_mlflow_env");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::create_dir_all(temp_dir.join("mlruns")).unwrap();

        let item = check_negative_result_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-09");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_cargo_and_seed() {
        let temp_dir = std::env::temp_dir().join("test_seed_env");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create a Rust file with seed constant
        std::fs::write(temp_dir.join("src/lib.rs"), r#"const SEED: u64 = 42;"#).unwrap();

        let item = check_random_seed_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-04");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_all_severities_present() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);

        let has_major = items.iter().any(|i| i.severity == Severity::Major);
        let has_critical = items.iter().any(|i| i.severity == Severity::Critical);

        assert!(
            has_major || has_critical,
            "Expected at least one Major or Critical check"
        );
    }

    // =========================================================================
    // Additional Coverage Tests for Branch Coverage
    // =========================================================================

    #[test]
    fn test_hdd_cov_001_hypothesis_with_pr_template() {
        let temp_dir = std::env::temp_dir().join("test_pr_template");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".github")).unwrap();

        // Create PR template with hypothesis section
        std::fs::write(
            temp_dir.join(".github/PULL_REQUEST_TEMPLATE.md"),
            "## Hypothesis\n\nDescribe your hypothesis here\n",
        )
        .unwrap();

        let item = check_hypothesis_statement(&temp_dir);
        assert_eq!(item.id, "HDD-01");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_002_hypothesis_template_without_section() {
        let temp_dir = std::env::temp_dir().join("test_pr_no_hypo");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".github")).unwrap();

        // Create PR template without hypothesis section
        std::fs::write(
            temp_dir.join(".github/PULL_REQUEST_TEMPLATE.md"),
            "## Description\n\nDescribe your changes\n",
        )
        .unwrap();

        let item = check_hypothesis_statement(&temp_dir);
        assert_eq!(item.id, "HDD-01");
        // Should be partial since template exists but no hypothesis section

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_003_baseline_with_benchmarks() {
        let temp_dir = std::env::temp_dir().join("test_benches");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("benches")).unwrap();
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create a benchmark file and baseline test
        std::fs::write(temp_dir.join("benches/bench.rs"), "#[bench]\nfn bench() {}").unwrap();
        std::fs::write(temp_dir.join("src/lib.rs"), "fn compare_to_baseline() {}").unwrap();

        let item = check_baseline_comparison(&temp_dir);
        assert_eq!(item.id, "HDD-02");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_004_reproducibility_with_makefile() {
        let temp_dir = std::env::temp_dir().join("test_reproduce");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".github/workflows")).unwrap();

        std::fs::write(temp_dir.join("Makefile"), "reproduce:\n\techo reproduce").unwrap();
        std::fs::write(
            temp_dir.join(".github/workflows/ci.yml"),
            "name: CI\njobs:\n  reproduce:\n    runs-on: ubuntu-latest",
        )
        .unwrap();

        let item = check_gold_reproducibility(&temp_dir);
        assert_eq!(item.id, "HDD-03");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_005_seed_with_rng() {
        let temp_dir = std::env::temp_dir().join("test_seeded_rng");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
use rand::SeedableRng;
const RANDOM_SEED: u64 = 42;
fn create_rng() -> StdRng {
    StdRng::seed_from_u64(RANDOM_SEED)
}
"#,
        )
        .unwrap();

        let item = check_random_seed_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-04");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_006_containerization_with_nix() {
        let temp_dir = std::env::temp_dir().join("test_nix");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(temp_dir.join("Cargo.lock"), "# cargo lock").unwrap();
        std::fs::write(temp_dir.join("flake.nix"), "{ inputs = {}; }").unwrap();

        let item = check_environment_containerization(&temp_dir);
        assert_eq!(item.id, "HDD-05");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_007_data_version_with_dvc() {
        let temp_dir = std::env::temp_dir().join("test_dvc_yaml");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(temp_dir.join("dvc.yaml"), "stages:\n  train:").unwrap();
        std::fs::write(temp_dir.join("src/lib.rs"), "fn load_data() {}").unwrap();

        let item = check_data_version_control(&temp_dir);
        assert_eq!(item.id, "HDD-06");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_008_stats_with_effect_size() {
        let temp_dir = std::env::temp_dir().join("test_stats");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
fn accuracy() -> f64 { 0.95 }
fn p_value() -> f64 { 0.01 }
fn effect_size() -> f64 { 0.8 }
fn cohen_d(a: &[f64], b: &[f64]) -> f64 { 0.0 }
"#,
        )
        .unwrap();

        let item = check_statistical_significance(&temp_dir);
        assert_eq!(item.id, "HDD-07");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_009_ablation_study() {
        let temp_dir = std::env::temp_dir().join("test_ablation");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
mod neural;
fn ablation() {}
fn transformer() {}
"#,
        )
        .unwrap();

        let item = check_ablation_study(&temp_dir);
        assert_eq!(item.id, "HDD-08");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_010_negative_docs_with_adr() {
        let temp_dir = std::env::temp_dir().join("test_adr");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("docs/adr")).unwrap();

        std::fs::write(
            temp_dir.join("docs/adr/001-failed.md"),
            "# Decision: Did not work\n",
        )
        .unwrap();

        let item = check_negative_result_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-09");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_011_metric_prereg() {
        let temp_dir = std::env::temp_dir().join("test_prereg");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "const PRE_REGISTRATION: &str = \"metrics defined\";\n",
        )
        .unwrap();

        let item = check_metric_preregistration(&temp_dir);
        assert_eq!(item.id, "HDD-10");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_edd_cov_001_equation_with_emc() {
        let temp_dir = std::env::temp_dir().join("test_emc");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("docs/emc")).unwrap();
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Must have both EMC and math docs (derivation, proof, analytical_solution, LaTeX)
        std::fs::write(
            temp_dir.join("docs/emc/equation.md"),
            "# Governing Equation\n\n## Derivation\n\n## Proof\n",
        )
        .unwrap();
        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "fn simulate() {}\n// derivation included",
        )
        .unwrap();

        let item = check_equation_verification(&temp_dir);
        assert_eq!(item.id, "EDD-01");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_edd_cov_002_emc_completeness_full() {
        let temp_dir = std::env::temp_dir().join("test_emc_full");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("docs/emc")).unwrap();
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("docs/emc/model.md"),
            r#"# Model
## Governing Equations
## Validity Domain
## Derivation
## Stability Analysis
## Verification Tests
"#,
        )
        .unwrap();
        std::fs::write(temp_dir.join("src/lib.rs"), "fn simulate() {}").unwrap();

        let item = check_emc_completeness(&temp_dir);
        assert_eq!(item.id, "EDD-02");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_edd_cov_003_numerical_validation() {
        let temp_dir = std::env::temp_dir().join("test_numerical");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
fn simulate() {}
fn analytical_solution(t: f64) -> f64 { t.exp() }
const TOLERANCE: f64 = 1e-6;
fn assert_relative_eq(a: f64, b: f64) {}
"#,
        )
        .unwrap();

        let item = check_numerical_analytical_validation(&temp_dir);
        assert_eq!(item.id, "EDD-03");
        assert!(matches!(item.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_012_ci_pattern_gitlab() {
        let temp_dir = std::env::temp_dir().join("test_gitlab_ci");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(temp_dir.join(".gitlab-ci.yml"), "stages:\n  - reproduce\n").unwrap();

        let result = check_ci_for_pattern(&temp_dir, &["reproduce"]);
        assert!(result);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_013_no_random_usage() {
        let temp_dir = std::env::temp_dir().join("test_no_random");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // No random usage at all
        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "fn deterministic() -> i32 { 42 }",
        )
        .unwrap();

        let item = check_random_seed_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-04");
        assert!(matches!(item.status, CheckStatus::Pass)); // Should pass if no randomness used

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_014_lock_only_no_container() {
        let temp_dir = std::env::temp_dir().join("test_lock_only");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(temp_dir.join("Cargo.lock"), "# lock file").unwrap();
        // No Dockerfile or nix

        let item = check_environment_containerization(&temp_dir);
        assert_eq!(item.id, "HDD-05");
        // Should be partial - locked but no container

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_015_docker_compose() {
        let temp_dir = std::env::temp_dir().join("test_compose");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(temp_dir.join("Cargo.lock"), "# lock").unwrap();
        std::fs::write(
            temp_dir.join("docker-compose.yaml"),
            "version: '3'\nservices:",
        )
        .unwrap();

        let item = check_environment_containerization(&temp_dir);
        assert_eq!(item.id, "HDD-05");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_016_check_pattern_config_files() {
        let temp_dir = std::env::temp_dir().join("test_pattern_config");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create config with pattern
        std::fs::write(
            temp_dir.join("config.yaml"),
            "metrics:\n  - accuracy\n  - F1\n",
        )
        .unwrap();

        let result = check_for_pattern(&temp_dir, &["metrics:"]);
        assert!(result);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_017_experiments_dir() {
        let temp_dir = std::env::temp_dir().join("test_experiments");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("experiments")).unwrap();

        let item = check_negative_result_documentation(&temp_dir);
        assert_eq!(item.id, "HDD-09");
        // Should be partial - experiments dir exists

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hdd_cov_018_sensitivity_no_ablation() {
        let temp_dir = std::env::temp_dir().join("test_sensitivity");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "fn neural() {}\nfn grid_search() {}\n",
        )
        .unwrap();

        let item = check_ablation_study(&temp_dir);
        assert_eq!(item.id, "HDD-08");
        // Should be partial - sensitivity but no formal ablation

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_edd_cov_004_partial_emc() {
        let temp_dir = std::env::temp_dir().join("test_partial_emc");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "fn simulate() {}\n// governing equation",
        )
        .unwrap();

        let item = check_equation_verification(&temp_dir);
        assert_eq!(item.id, "EDD-01");
        // Should be partial - has simulation and some docs

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
