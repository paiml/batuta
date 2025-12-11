//! Section 3: Hypothesis-Driven & Equation-Driven Development (HDD-01 to EDD-03)
//!
//! Implements Scientific Method integration for ML development.
//!
//! # TPS Principles
//!
//! - **Scientific Method**: Falsifiable hypotheses, pre-registration
//! - **Muda**: Overprocessing prevention via baseline comparison
//! - **Kaizen**: Learning from negative results

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
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

    if has_hypothesis_section {
        item = item.pass();
    } else if has_pr_template || has_hypothesis_docs {
        item = item.partial("PR template exists but missing hypothesis section");
    } else {
        // Check if project does ML
        let is_ml_project = check_for_pattern(project_path, &["model", "train", "predict"]);
        if !is_ml_project {
            item = item.pass(); // Not an ML project
        } else {
            item = item.partial("No hypothesis requirement in PR workflow");
        }
    }

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

    if has_baseline_tests && has_benchmarks {
        item = item.pass();
    } else if has_baseline_tests || has_comparison_docs {
        item = item.partial("Some baseline comparison infrastructure");
    } else {
        let is_ml = check_for_pattern(project_path, &["neural", "transformer", "deep_learning"]);
        if !is_ml {
            item = item.pass(); // Not a complex ML project
        } else {
            item = item.partial("Complex models without baseline comparison");
        }
    }

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

    if has_reproduce_target && has_ci_repro {
        item = item.pass();
    } else if has_reproduce_target || has_repro_docs {
        item = item.partial("Reproduction target exists (not in CI)");
    } else {
        let is_ml = check_for_pattern(project_path, &["train", "model", "weights"]);
        if !is_ml {
            item = item.pass(); // Not an ML project
        } else {
            item = item.partial("No reproduction infrastructure");
        }
    }

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

    // Check if project uses randomness
    let uses_random = check_for_pattern(project_path, &["rand::", "random", "Rng", "thread_rng"]);

    if !uses_random {
        item = item.pass(); // No randomness
    } else if has_seeded_rng && has_seed_pinning {
        item = item.pass();
    } else if has_seed_pinning {
        item = item.partial("Seed usage found (verify documentation)");
    } else {
        item = item.partial("Random usage without explicit seed pinning");
    }

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

    if (has_dockerfile || has_nix) && has_lock_file {
        item = item.pass();
    } else if has_lock_file {
        item = item.partial("Dependencies locked but no containerization");
    } else {
        item = item.partial("No environment containerization");
    }

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

    // Check if project uses external data
    let uses_data = check_for_pattern(project_path, &["load_data", "Dataset", "DataLoader"]);

    if !uses_data {
        item = item.pass(); // No external data
    } else if has_dvc {
        item = item.pass();
    } else if has_data_versioning {
        item = item.partial("Data versioning patterns found (no DVC)");
    } else {
        item = item.partial("Data handling without version control");
    }

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

    // Check if project makes performance claims
    let has_perf_claims =
        check_for_pattern(project_path, &["accuracy", "F1", "precision", "recall"]);

    if !has_perf_claims {
        item = item.pass(); // No performance claims
    } else if has_stats && has_effect_size {
        item = item.pass();
    } else if has_stats {
        item = item.partial("Statistical testing (missing effect size)");
    } else {
        item = item.partial("Performance metrics without significance testing");
    }

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

    // Check if project has complex models
    let has_complex_models = check_for_pattern(
        project_path,
        &["neural", "transformer", "ensemble", "multi_layer"],
    );

    if !has_complex_models {
        item = item.pass(); // No complex models
    } else if has_ablation {
        item = item.pass();
    } else if has_sensitivity {
        item = item.partial("Sensitivity analysis (no formal ablation)");
    } else {
        item = item.partial("Complex models without ablation studies");
    }

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

    if has_negative_docs || has_adr {
        item = item.pass();
    } else if has_experiment_log {
        item = item.partial("Experiment logging (check for negative results)");
    } else {
        item = item.partial("No negative result documentation");
    }

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

    if has_prereg {
        item = item.pass();
    } else if has_metric_config {
        item = item.partial("Metrics in config (verify pre-registration)");
    } else {
        let is_ml = check_for_pattern(project_path, &["accuracy", "loss", "evaluate"]);
        if !is_ml {
            item = item.pass();
        } else {
            item = item.partial("No metric pre-registration");
        }
    }

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

    if !has_simulation {
        item = item.pass(); // No simulation code
    } else if has_emc && has_math_docs {
        item = item.pass();
    } else if has_emc || has_math_docs {
        item = item.partial("Partial equation documentation");
    } else {
        item = item.partial("Simulation without equation verification");
    }

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

    if !has_simulation {
        item = item.pass(); // No simulation
    } else if has_emc_dir && sections_found >= 4 {
        item = item.pass();
    } else if sections_found >= 2 {
        item = item.partial("Partial EMC documentation");
    } else {
        item = item.partial("Missing EMC documentation");
    }

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

    if !has_simulation {
        item = item.pass(); // No numerical simulation
    } else if has_analytical_tests && has_tolerance {
        item = item.pass();
    } else if has_verification || has_analytical_tests {
        item = item.partial("Some validation (verify tolerance)");
    } else {
        item = item.partial("Numerical code without analytical validation");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for pattern in patterns {
                    if content.contains(pattern) {
                        return true;
                    }
                }
            }
        }
    }

    // Also check config and doc files
    let extensions = ["yaml", "toml", "json", "md"];
    for ext in extensions {
        if let Ok(entries) = glob::glob(&format!("{}/**/*.{}", project_path.display(), ext)) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(&entry) {
                    for pattern in patterns {
                        if content.contains(pattern) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

fn check_ci_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    let ci_paths = [
        format!("{}/.github/workflows/*.yml", project_path.display()),
        format!("{}/.github/workflows/*.yaml", project_path.display()),
        format!("{}/.gitlab-ci.yml", project_path.display()),
    ];

    for glob_pattern in &ci_paths {
        if let Ok(entries) = glob::glob(glob_pattern) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(&entry) {
                    for pattern in patterns {
                        if content.contains(pattern) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
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
}
