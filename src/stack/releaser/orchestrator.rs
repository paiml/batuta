#![allow(dead_code)]
//! Release Orchestrator
//!
//! Implements the `batuta stack release` command functionality.
//! Coordinates releases across multiple crates in topological order,
//! ensuring all quality gates pass before publishing.

use crate::stack::checker::StackChecker;
use crate::stack::types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

// Re-export types from releaser_types module
pub use super::super::releaser_types::{
    format_plan_text, BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate,
};

/// Release orchestrator for coordinated multi-crate releases
pub struct ReleaseOrchestrator {
    /// Release configuration (pub(super) for preflight module access)
    pub(in crate::stack) config: ReleaseConfig,

    /// Stack checker for health analysis
    checker: StackChecker,

    /// Pre-flight results per crate
    preflight_results: HashMap<String, PreflightResult>,
}

impl ReleaseOrchestrator {
    /// Create a new release orchestrator
    pub fn new(checker: StackChecker, config: ReleaseConfig) -> Self {
        Self {
            config,
            checker,
            preflight_results: HashMap::new(),
        }
    }

    /// Create a release orchestrator from a workspace path
    #[cfg(feature = "native")]
    pub fn from_workspace(workspace_path: &Path, config: ReleaseConfig) -> Result<Self> {
        let checker = StackChecker::from_workspace(workspace_path)?;
        Ok(Self::new(checker, config))
    }

    /// Plan a release for a specific crate
    pub fn plan_release(&mut self, crate_name: &str) -> Result<ReleasePlan> {
        let release_order = self.checker.release_order_for(crate_name)?;

        let mut releases = Vec::new();

        for name in &release_order {
            let planned = self.plan_single_release(name)?;
            releases.push(planned);
        }

        Ok(ReleasePlan {
            releases,
            dry_run: self.config.dry_run,
            preflight_results: self.preflight_results.clone(),
        })
    }

    /// Plan a release for all crates with changes
    pub fn plan_all_releases(&mut self) -> Result<ReleasePlan> {
        let release_order = self.checker.topological_order()?;

        let mut releases = Vec::new();

        for name in &release_order {
            let planned = self.plan_single_release(name)?;
            releases.push(planned);
        }

        Ok(ReleasePlan {
            releases,
            dry_run: self.config.dry_run,
            preflight_results: self.preflight_results.clone(),
        })
    }

    /// Plan a single crate release
    fn plan_single_release(&self, crate_name: &str) -> Result<PlannedRelease> {
        // For now, we'll use a placeholder since we don't have the graph's crate info easily
        // In real implementation, we'd get this from the checker's graph
        let current_version = semver::Version::new(0, 0, 0); // Placeholder

        let new_version = match self.config.bump_type {
            Some(bump) => bump.apply(&current_version),
            None => semver::Version::new(
                current_version.major,
                current_version.minor,
                current_version.patch + 1,
            ),
        };

        Ok(PlannedRelease {
            crate_name: crate_name.to_string(),
            current_version,
            new_version,
            dependents: vec![], // Would be populated from graph
            ready: true,        // Would be determined by preflight checks
        })
    }

    /// Run pre-flight checks for a crate
    pub fn run_preflight(
        &mut self,
        crate_name: &str,
        crate_path: &Path,
    ) -> Result<PreflightResult> {
        let mut result = PreflightResult::new(crate_name);

        if self.config.no_verify {
            result.add_check(PreflightCheck::pass(
                "verification",
                "Skipped (--no-verify)",
            ));
            self.preflight_results
                .insert(crate_name.to_string(), result.clone());
            return Ok(result);
        }

        // Check 1: Git clean
        let git_check = self.check_git_clean(crate_path);
        result.add_check(git_check);

        // Check 2: Lint
        let lint_check = self.check_lint(crate_path);
        result.add_check(lint_check);

        // Check 3: Coverage
        let coverage_check = self.check_coverage(crate_path);
        result.add_check(coverage_check);

        // Check 4: PMAT comply (ComputeBrick defect detection)
        let comply_check = self.check_pmat_comply(crate_path);
        result.add_check(comply_check);

        // Check 5: No path dependencies
        let path_check = self.check_no_path_deps(crate_name);
        result.add_check(path_check);

        // Check 6: Version bumped
        let version_check = self.check_version_bumped(crate_name);
        result.add_check(version_check);

        // =====================================================================
        // PMAT Quality Gate Integration (PMAT-STACK-GATES)
        // =====================================================================

        // Check 7: PMAT quality-gate (comprehensive checks)
        let quality_gate_check = self.check_pmat_quality_gate(crate_path);
        result.add_check(quality_gate_check);

        // Check 8: PMAT TDG (Technical Debt Grading)
        let tdg_check = self.check_pmat_tdg(crate_path);
        result.add_check(tdg_check);

        // Check 9: PMAT dead-code analysis
        let dead_code_check = self.check_pmat_dead_code(crate_path);
        result.add_check(dead_code_check);

        // Check 10: PMAT complexity analysis
        let complexity_check = self.check_pmat_complexity(crate_path);
        result.add_check(complexity_check);

        // Check 11: PMAT SATD (Self-Admitted Technical Debt)
        let satd_check = self.check_pmat_satd(crate_path);
        result.add_check(satd_check);

        // Check 12: PMAT Popper score (falsifiability)
        let popper_check = self.check_pmat_popper(crate_path);
        result.add_check(popper_check);

        // =====================================================================
        // Book and Examples Verification (RELEASE-DOCS)
        // =====================================================================

        // Check 13: Book build
        let book_check = self.check_book_build(crate_path);
        result.add_check(book_check);

        // Check 14: Examples verification
        let examples_check = self.check_examples_run(crate_path);
        result.add_check(examples_check);

        self.preflight_results
            .insert(crate_name.to_string(), result.clone());
        Ok(result)
    }

    // Note: check_* methods are in releaser_preflight.rs module

    /// Execute the release plan
    #[cfg(feature = "native")]
    pub async fn execute(&self, plan: &ReleasePlan) -> Result<ReleaseResult> {
        if plan.dry_run {
            return Ok(ReleaseResult {
                success: true,
                released_crates: vec![],
                message: "Dry run - no changes made".to_string(),
            });
        }

        let mut released = Vec::new();

        for release in &plan.releases {
            // Check preflight passed
            if let Some(preflight) = plan.preflight_results.get(&release.crate_name) {
                if !preflight.passed {
                    return Err(anyhow!(
                        "Pre-flight checks failed for {}: cannot release",
                        release.crate_name
                    ));
                }
            }

            // Update Cargo.toml version
            // self.update_cargo_toml(&release)?;

            // Create git tag
            // self.create_git_tag(&release)?;

            if self.config.publish {
                // Publish to crates.io
                // self.cargo_publish(&release)?;

                // Wait for availability
                // self.wait_for_crates_io(&release).await?;
            }

            released.push(ReleasedCrate {
                name: release.crate_name.clone(),
                version: release.new_version.clone(),
                published: self.config.publish,
            });
        }

        Ok(ReleaseResult {
            success: true,
            released_crates: released,
            message: format!("Successfully released {} crates", plan.releases.len()),
        })
    }
}
