//! Unit tests for Release Orchestrator
//!
//! Tests for ReleaseOrchestrator functionality.

use super::orchestrator::ReleaseOrchestrator;
use crate::stack::checker::StackChecker;
use crate::stack::graph::{DependencyEdge, DependencyGraph};
use crate::stack::releaser_types::{
    format_plan_text, BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate,
};
use crate::stack::types::*;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

// ============================================================================
// UNIT TESTS - Fast, focused, deterministic
// Following bashrs style: ARRANGE/ACT/ASSERT with task IDs
// ============================================================================

fn create_test_graph() -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    graph.add_crate(CrateInfo::new(
        "trueno",
        semver::Version::new(1, 2, 0),
        PathBuf::from("trueno/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "aprender",
        semver::Version::new(0, 8, 1),
        PathBuf::from("aprender/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "entrenar",
        semver::Version::new(0, 2, 2),
        PathBuf::from("entrenar/Cargo.toml"),
    ));

    graph.add_dependency(
        "aprender",
        "trueno",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    graph.add_dependency(
        "entrenar",
        "aprender",
        DependencyEdge {
            version_req: "^0.8".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    graph
}

#[test]
fn test_bump_type_patch() {
    let version = semver::Version::new(1, 2, 3);
    let bumped = BumpType::Patch.apply(&version);
    assert_eq!(bumped, semver::Version::new(1, 2, 4));
}

#[test]
fn test_bump_type_minor() {
    let version = semver::Version::new(1, 2, 3);
    let bumped = BumpType::Minor.apply(&version);
    assert_eq!(bumped, semver::Version::new(1, 3, 0));
}

#[test]
fn test_bump_type_major() {
    let version = semver::Version::new(1, 2, 3);
    let bumped = BumpType::Major.apply(&version);
    assert_eq!(bumped, semver::Version::new(2, 0, 0));
}

#[test]
fn test_release_config_default() {
    let config = ReleaseConfig::default();
    assert!(!config.no_verify);
    assert!(!config.dry_run);
    assert!(!config.publish);
    assert_eq!(config.min_coverage, 90.0);
}

#[test]
fn test_orchestrator_creation() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // Should be able to plan releases
    assert!(!orchestrator.config.dry_run);
}

#[test]
fn test_plan_release() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let plan = orchestrator.plan_release("entrenar").unwrap();

    // Should include all dependencies in order
    assert!(!plan.releases.is_empty());
    assert!(plan.dry_run);

    // entrenar should be last
    assert_eq!(plan.releases.last().unwrap().crate_name, "entrenar");
}

#[test]
fn test_plan_all_releases() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let plan = orchestrator.plan_all_releases().unwrap();

    // Should include all crates
    assert_eq!(plan.releases.len(), 3);
}

#[test]
fn test_preflight_no_verify() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        no_verify: true,
        ..Default::default()
    };
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let result = orchestrator
        .run_preflight("trueno", Path::new("."))
        .unwrap();

    // Should pass when no_verify is set
    assert!(result.passed);
    assert_eq!(result.checks.len(), 1);
    assert!(result.checks[0].message.contains("Skipped"));
}

#[test]
fn test_preflight_result_aggregation() {
    let mut result = PreflightResult::new("test");

    // All passing
    result.add_check(PreflightCheck::pass("check1", "ok"));
    result.add_check(PreflightCheck::pass("check2", "ok"));
    assert!(result.passed);

    // One failing
    result.add_check(PreflightCheck::fail("check3", "failed"));
    assert!(!result.passed);
}

#[test]
fn test_format_plan_text() {
    let plan = ReleasePlan {
        releases: vec![
            PlannedRelease {
                crate_name: "trueno".to_string(),
                current_version: semver::Version::new(1, 2, 0),
                new_version: semver::Version::new(1, 2, 1),
                dependents: vec![],
                ready: true,
            },
            PlannedRelease {
                crate_name: "aprender".to_string(),
                current_version: semver::Version::new(0, 8, 1),
                new_version: semver::Version::new(0, 8, 2),
                dependents: vec![],
                ready: true,
            },
        ],
        dry_run: true,
        preflight_results: HashMap::new(),
    };

    let text = format_plan_text(&plan);

    assert!(text.contains("DRY RUN"));
    assert!(text.contains("trueno"));
    assert!(text.contains("aprender"));
    assert!(text.contains("1.2.0 → 1.2.1"));
}

#[test]
fn test_released_crate() {
    let released = ReleasedCrate {
        name: "trueno".to_string(),
        version: semver::Version::new(1, 2, 1),
        published: true,
    };

    assert_eq!(released.name, "trueno");
    assert!(released.published);
}

#[test]
fn test_release_result() {
    let result = ReleaseResult {
        success: true,
        released_crates: vec![ReleasedCrate {
            name: "trueno".to_string(),
            version: semver::Version::new(1, 2, 1),
            published: true,
        }],
        message: "Success".to_string(),
    };

    assert!(result.success);
    assert_eq!(result.released_crates.len(), 1);
}

// ============================================================================
// RELEASE-001: BumpType edge cases
// ============================================================================

/// RED PHASE: Test BumpType::Patch on version 0.0.0
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_patch_from_zero() {
    // ARRANGE
    let version = semver::Version::new(0, 0, 0);

    // ACT
    let bumped = BumpType::Patch.apply(&version);

    // ASSERT
    assert_eq!(bumped, semver::Version::new(0, 0, 1));
}

/// RED PHASE: Test BumpType::Minor resets patch to 0
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_minor_resets_patch() {
    // ARRANGE
    let version = semver::Version::new(1, 2, 99);

    // ACT
    let bumped = BumpType::Minor.apply(&version);

    // ASSERT
    assert_eq!(bumped, semver::Version::new(1, 3, 0));
}

/// RED PHASE: Test BumpType::Major resets minor and patch
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_major_resets_minor_patch() {
    // ARRANGE
    let version = semver::Version::new(5, 99, 99);

    // ACT
    let bumped = BumpType::Major.apply(&version);

    // ASSERT
    assert_eq!(bumped, semver::Version::new(6, 0, 0));
}

/// RED PHASE: Test BumpType equality
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_type_equality() {
    assert_eq!(BumpType::Patch, BumpType::Patch);
    assert_eq!(BumpType::Minor, BumpType::Minor);
    assert_eq!(BumpType::Major, BumpType::Major);
    assert_ne!(BumpType::Patch, BumpType::Minor);
}

/// RED PHASE: Test BumpType clone
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_type_clone() {
    let bump = BumpType::Minor;
    let cloned = bump;
    assert_eq!(bump, cloned);
}

/// RED PHASE: Test BumpType debug
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_001_bump_type_debug() {
    assert!(format!("{:?}", BumpType::Patch).contains("Patch"));
    assert!(format!("{:?}", BumpType::Minor).contains("Minor"));
    assert!(format!("{:?}", BumpType::Major).contains("Major"));
}

// ============================================================================
// RELEASE-002: ReleaseConfig variations
// ============================================================================

/// RED PHASE: Test ReleaseConfig with custom values
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_002_config_custom_values() {
    // ARRANGE & ACT
    let config = ReleaseConfig {
        bump_type: Some(BumpType::Minor),
        no_verify: true,
        dry_run: true,
        publish: true,
        min_coverage: 95.0,
        lint_command: "cargo clippy".to_string(),
        coverage_command: "cargo tarpaulin".to_string(),
        comply_command: "pmat comply --strict".to_string(),
        fail_on_comply_violations: true,
        ..Default::default()
    };

    // ASSERT
    assert!(config.no_verify);
    assert!(config.dry_run);
    assert!(config.publish);
    assert_eq!(config.min_coverage, 95.0);
    assert_eq!(config.lint_command, "cargo clippy");
    assert_eq!(config.bump_type, Some(BumpType::Minor));
    assert_eq!(config.comply_command, "pmat comply --strict");
    assert!(config.fail_on_comply_violations);
}

/// RED PHASE: Test ReleaseConfig clone
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_002_config_clone() {
    let config = ReleaseConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.min_coverage, config.min_coverage);
    assert_eq!(cloned.dry_run, config.dry_run);
}

/// RED PHASE: Test ReleaseConfig debug
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_002_config_debug() {
    let config = ReleaseConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ReleaseConfig"));
    assert!(debug.contains("min_coverage"));
}

// ============================================================================
// RELEASE-003: format_plan_text variations
// ============================================================================

/// RED PHASE: Test format_plan_text without dry run
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_003_format_plan_text_live() {
    // ARRANGE
    let plan = ReleasePlan {
        releases: vec![PlannedRelease {
            crate_name: "test-crate".to_string(),
            current_version: semver::Version::new(1, 0, 0),
            new_version: semver::Version::new(1, 0, 1),
            dependents: vec![],
            ready: true,
        }],
        dry_run: false,
        preflight_results: HashMap::new(),
    };

    // ACT
    let text = format_plan_text(&plan);

    // ASSERT
    assert!(!text.contains("DRY RUN"));
    assert!(text.contains("Release Plan"));
    assert!(text.contains("test-crate"));
}

/// RED PHASE: Test format_plan_text with preflight results
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_003_format_plan_text_with_preflight() {
    // ARRANGE
    let mut preflight_results = HashMap::new();
    preflight_results.insert(
        "trueno".to_string(),
        PreflightResult {
            crate_name: "trueno".to_string(),
            checks: vec![PreflightCheck::pass("git", "clean")],
            passed: true,
        },
    );
    preflight_results.insert(
        "aprender".to_string(),
        PreflightResult {
            crate_name: "aprender".to_string(),
            checks: vec![PreflightCheck::fail("lint", "errors")],
            passed: false,
        },
    );

    let plan = ReleasePlan {
        releases: vec![
            PlannedRelease {
                crate_name: "trueno".to_string(),
                current_version: semver::Version::new(1, 0, 0),
                new_version: semver::Version::new(1, 0, 1),
                dependents: vec![],
                ready: true,
            },
            PlannedRelease {
                crate_name: "aprender".to_string(),
                current_version: semver::Version::new(0, 8, 0),
                new_version: semver::Version::new(0, 8, 1),
                dependents: vec![],
                ready: false,
            },
        ],
        dry_run: true,
        preflight_results,
    };

    // ACT
    let text = format_plan_text(&plan);

    // ASSERT
    assert!(text.contains("trueno"));
    assert!(text.contains("aprender"));
}

/// RED PHASE: Test format_plan_text empty plan
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_003_format_plan_text_empty() {
    // ARRANGE
    let plan = ReleasePlan {
        releases: vec![],
        dry_run: false,
        preflight_results: HashMap::new(),
    };

    // ACT
    let text = format_plan_text(&plan);

    // ASSERT
    assert!(text.contains("Release Plan"));
    assert!(text.contains("Release order"));
}

// ============================================================================
// RELEASE-004: PlannedRelease and ReleasePlan
// ============================================================================

/// RED PHASE: Test PlannedRelease with dependents
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_004_planned_release_with_dependents() {
    // ARRANGE & ACT
    let release = PlannedRelease {
        crate_name: "trueno".to_string(),
        current_version: semver::Version::new(1, 0, 0),
        new_version: semver::Version::new(1, 1, 0),
        dependents: vec!["aprender".to_string(), "trueno-db".to_string()],
        ready: true,
    };

    // ASSERT
    assert_eq!(release.dependents.len(), 2);
    assert!(release.dependents.contains(&"aprender".to_string()));
    assert!(release.ready);
}

/// RED PHASE: Test ReleasePlan dry_run flag
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_004_release_plan_dry_run() {
    let plan_dry = ReleasePlan {
        releases: vec![],
        dry_run: true,
        preflight_results: HashMap::new(),
    };

    let plan_live = ReleasePlan {
        releases: vec![],
        dry_run: false,
        preflight_results: HashMap::new(),
    };

    assert!(plan_dry.dry_run);
    assert!(!plan_live.dry_run);
}

// ============================================================================
// RELEASE-005: ReleaseResult and ReleasedCrate
// ============================================================================

/// RED PHASE: Test ReleaseResult failure
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_005_release_result_failure() {
    let result = ReleaseResult {
        success: false,
        released_crates: vec![],
        message: "Pre-flight checks failed".to_string(),
    };

    assert!(!result.success);
    assert!(result.released_crates.is_empty());
    assert!(result.message.contains("failed"));
}

/// RED PHASE: Test ReleasedCrate unpublished
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_005_released_crate_unpublished() {
    let released = ReleasedCrate {
        name: "local-only".to_string(),
        version: semver::Version::new(0, 1, 0),
        published: false,
    };

    assert!(!released.published);
    assert_eq!(released.name, "local-only");
}

/// RED PHASE: Test ReleaseResult debug
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_005_release_result_debug() {
    let result = ReleaseResult {
        success: true,
        released_crates: vec![],
        message: "OK".to_string(),
    };

    let debug = format!("{:?}", result);
    assert!(debug.contains("ReleaseResult"));
    assert!(debug.contains("success"));
}

/// RED PHASE: Test ReleasedCrate debug
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_005_released_crate_debug() {
    let released = ReleasedCrate {
        name: "test".to_string(),
        version: semver::Version::new(1, 0, 0),
        published: true,
    };

    let debug = format!("{:?}", released);
    assert!(debug.contains("ReleasedCrate"));
    assert!(debug.contains("test"));
}

// ============================================================================
// RELEASE-006: Orchestrator edge cases
// ============================================================================

/// RED PHASE: Test plan_single_release with bump_type
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_006_plan_with_bump_type() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        bump_type: Some(BumpType::Minor),
        dry_run: true,
        ..Default::default()
    };
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let plan = orchestrator.plan_release("trueno").unwrap();

    // All releases should use Minor bump
    for release in &plan.releases {
        // New version should have patch = 0 (minor bump)
        assert_eq!(release.new_version.patch, 0);
    }
}

/// RED PHASE: Test plan_release for leaf crate (no dependencies)
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_006_plan_leaf_crate() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    // trueno has no dependencies, should be first in release order
    let result = orchestrator.plan_release("trueno");
    assert!(result.is_ok());

    let plan = result.unwrap();
    assert_eq!(plan.releases.len(), 1);
    assert_eq!(plan.releases[0].crate_name, "trueno");
}

// ============================================================================
// RELEASE-007: Preflight checks
// ============================================================================

/// RED PHASE: Test preflight with no_verify skips checks
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_007_preflight_no_verify() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        no_verify: true,
        ..Default::default()
    };
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let result = orchestrator.run_preflight("trueno", std::path::Path::new("."));
    assert!(result.is_ok());

    let preflight = result.unwrap();
    assert!(preflight.passed);
    assert_eq!(preflight.checks.len(), 1);
    assert!(preflight.checks[0].message.contains("Skipped"));
}

/// RED PHASE: Test check_no_path_deps always passes (placeholder)
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_007_check_no_path_deps() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let check = orchestrator.check_no_path_deps("any-crate");
    assert!(check.passed);
    assert!(check.message.contains("No path dependencies"));
}

/// RED PHASE: Test check_version_bumped always passes (placeholder)
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_007_check_version_bumped() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let check = orchestrator.check_version_bumped("any-crate");
    assert!(check.passed);
    assert!(check.message.contains("ahead"));
}

// ============================================================================
// RELEASE-008: Execute function
// ============================================================================

/// RED PHASE: Test execute dry run returns success
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_008_execute_dry_run() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        dry_run: true,
        ..Default::default()
    };
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let plan = ReleasePlan {
        releases: vec![PlannedRelease {
            crate_name: "test".to_string(),
            current_version: semver::Version::new(1, 0, 0),
            new_version: semver::Version::new(1, 0, 1),
            dependents: vec![],
            ready: true,
        }],
        dry_run: true,
        preflight_results: HashMap::new(),
    };

    let result = orchestrator.execute(&plan);

    assert!(result.is_ok());
    let release_result = result.unwrap();
    assert!(release_result.success);
    assert!(release_result.released_crates.is_empty());
    assert!(release_result.message.contains("Dry run"));
}

/// RED PHASE: Test execute fails when preflight failed
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_008_execute_preflight_failed() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let mut preflight_results = HashMap::new();
    preflight_results.insert(
        "test".to_string(),
        PreflightResult {
            crate_name: "test".to_string(),
            checks: vec![PreflightCheck::fail("lint", "errors")],
            passed: false,
        },
    );

    let plan = ReleasePlan {
        releases: vec![PlannedRelease {
            crate_name: "test".to_string(),
            current_version: semver::Version::new(1, 0, 0),
            new_version: semver::Version::new(1, 0, 1),
            dependents: vec![],
            ready: false,
        }],
        dry_run: false,
        preflight_results,
    };

    let result = orchestrator.execute(&plan);

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Pre-flight checks failed"));
}

/// RED PHASE: Test execute success without publish
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_008_execute_success_no_publish() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        publish: false,
        ..Default::default()
    };
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let mut preflight_results = HashMap::new();
    preflight_results.insert(
        "test".to_string(),
        PreflightResult {
            crate_name: "test".to_string(),
            checks: vec![PreflightCheck::pass("all", "ok")],
            passed: true,
        },
    );

    let plan = ReleasePlan {
        releases: vec![PlannedRelease {
            crate_name: "test".to_string(),
            current_version: semver::Version::new(1, 0, 0),
            new_version: semver::Version::new(1, 0, 1),
            dependents: vec![],
            ready: true,
        }],
        dry_run: false,
        preflight_results,
    };

    let result = orchestrator.execute(&plan);

    assert!(result.is_ok());
    let release_result = result.unwrap();
    assert!(release_result.success);
    assert_eq!(release_result.released_crates.len(), 1);
    assert!(!release_result.released_crates[0].published);
}

/// RED PHASE: Test execute with multiple crates (fake paths = no publish)
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_008_execute_multiple_crates() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        publish: true,
        ..Default::default()
    };
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    let mut preflight_results = HashMap::new();
    for name in &["trueno", "aprender"] {
        preflight_results.insert(
            name.to_string(),
            PreflightResult {
                crate_name: name.to_string(),
                checks: vec![PreflightCheck::pass("all", "ok")],
                passed: true,
            },
        );
    }

    let plan = ReleasePlan {
        releases: vec![
            PlannedRelease {
                crate_name: "trueno".to_string(),
                current_version: semver::Version::new(1, 0, 0),
                new_version: semver::Version::new(1, 0, 1),
                dependents: vec!["aprender".to_string()],
                ready: true,
            },
            PlannedRelease {
                crate_name: "aprender".to_string(),
                current_version: semver::Version::new(0, 8, 0),
                new_version: semver::Version::new(0, 8, 1),
                dependents: vec![],
                ready: true,
            },
        ],
        dry_run: false,
        preflight_results,
    };

    let result = orchestrator.execute(&plan);

    assert!(result.is_ok());
    let release_result = result.unwrap();
    assert!(release_result.success);
    assert_eq!(release_result.released_crates.len(), 2);
    // Fake paths don't exist, so no actual publish occurs
    assert!(!release_result.released_crates[0].published);
    assert!(!release_result.released_crates[1].published);
}

// ============================================================================
// RELEASE-009: plan_all_releases
// ============================================================================

/// RED PHASE: Test plan_all_releases
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_009_plan_all_releases() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let result = orchestrator.plan_all_releases();
    assert!(result.is_ok());

    let plan = result.unwrap();
    // Should have all 3 crates from test graph
    assert!(!plan.releases.is_empty());
}

// ============================================================================
// RELEASE-010: ReleaseConfig variations
// ============================================================================

/// RED PHASE: Test ReleaseConfig with publish enabled
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_010_config_publish() {
    let config = ReleaseConfig {
        publish: true,
        dry_run: false,
        ..Default::default()
    };

    assert!(config.publish);
    assert!(!config.dry_run);
}

/// RED PHASE: Test ReleaseConfig lint command
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_010_config_lint_command() {
    let config = ReleaseConfig {
        lint_command: "cargo clippy -- -D warnings".to_string(),
        ..Default::default()
    };

    assert_eq!(config.lint_command, "cargo clippy -- -D warnings");
}

/// RED PHASE: Test ReleaseConfig coverage command
#[test]
#[allow(non_snake_case)]
fn test_RELEASE_010_config_coverage_command() {
    let config = ReleaseConfig {
        coverage_command: "cargo tarpaulin".to_string(),
        min_coverage: 95.0,
        ..Default::default()
    };

    assert_eq!(config.coverage_command, "cargo tarpaulin");
    assert_eq!(config.min_coverage, 95.0);
}

// ============================================================================
// RELEASE-DOCS: Book and Examples Verification Tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_config_defaults() {
    // ARRANGE
    let config = ReleaseConfig::default();

    // ASSERT - verify book and examples defaults
    assert_eq!(config.book_command, "mdbook build book");
    assert!(config.fail_on_book);
    assert_eq!(config.examples_command, "cargo run --example");
    assert!(config.fail_on_examples);
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_config_custom_book() {
    // ARRANGE/ACT
    let config = ReleaseConfig {
        book_command: "mdbook build docs/book".to_string(),
        fail_on_book: false,
        ..Default::default()
    };

    // ASSERT
    assert_eq!(config.book_command, "mdbook build docs/book");
    assert!(!config.fail_on_book);
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_config_custom_examples() {
    // ARRANGE/ACT
    let config = ReleaseConfig {
        examples_command: "cargo run --release --example".to_string(),
        fail_on_examples: false,
        ..Default::default()
    };

    // ASSERT
    assert_eq!(config.examples_command, "cargo run --release --example");
    assert!(!config.fail_on_examples);
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_config_disabled() {
    // ARRANGE/ACT - disable book and examples checks
    let config = ReleaseConfig {
        book_command: String::new(),
        examples_command: String::new(),
        ..Default::default()
    };

    // ASSERT
    assert!(config.book_command.is_empty());
    assert!(config.examples_command.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_check_book_no_command() {
    // ARRANGE
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        book_command: String::new(),
        ..Default::default()
    };
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // ACT
    let check = orchestrator.check_book_build(Path::new("."));

    // ASSERT
    assert!(check.passed);
    assert!(check.message.contains("skipped"));
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_check_book_no_directory() {
    // ARRANGE
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // ACT - check in a directory without a book folder
    let check = orchestrator.check_book_build(Path::new("/tmp"));

    // ASSERT
    assert!(check.passed);
    assert!(check.message.contains("No book directory"));
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_check_examples_no_command() {
    // ARRANGE
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig {
        examples_command: String::new(),
        ..Default::default()
    };
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // ACT
    let check = orchestrator.check_examples_run(Path::new("."));

    // ASSERT
    assert!(check.passed);
    assert!(check.message.contains("skipped"));
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_check_examples_no_directory() {
    // ARRANGE
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // ACT - check in a directory without an examples folder
    let check = orchestrator.check_examples_run(Path::new("/tmp"));

    // ASSERT
    assert!(check.passed);
    assert!(check.message.contains("No examples directory"));
}

#[test]
#[allow(non_snake_case)]
fn test_RELEASE_DOCS_discover_examples_empty() {
    // ARRANGE
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    let config = ReleaseConfig::default();
    let orchestrator = ReleaseOrchestrator::new(checker, config);

    // ACT - discover examples in a directory without any
    let examples = orchestrator.discover_examples(Path::new("/tmp"));

    // ASSERT
    assert!(examples.is_empty());
}
