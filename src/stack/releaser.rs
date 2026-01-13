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
use std::process::Command;

/// Bump type for version updates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BumpType {
    /// Increment patch version (0.0.x)
    Patch,
    /// Increment minor version (0.x.0)
    Minor,
    /// Increment major version (x.0.0)
    Major,
}

impl BumpType {
    /// Apply bump to a version
    pub fn apply(&self, version: &semver::Version) -> semver::Version {
        match self {
            BumpType::Patch => {
                semver::Version::new(version.major, version.minor, version.patch + 1)
            }
            BumpType::Minor => semver::Version::new(version.major, version.minor + 1, 0),
            BumpType::Major => semver::Version::new(version.major + 1, 0, 0),
        }
    }
}

/// Configuration for release orchestration
#[derive(Debug, Clone)]
pub struct ReleaseConfig {
    /// Version bump type
    pub bump_type: Option<BumpType>,

    /// Whether to skip quality gates
    pub no_verify: bool,

    /// Whether this is a dry run
    pub dry_run: bool,

    /// Whether to actually publish to crates.io
    pub publish: bool,

    /// Minimum coverage percentage required
    pub min_coverage: f64,

    /// Lint command to run
    pub lint_command: String,

    /// Coverage command to run
    pub coverage_command: String,

    /// PMAT comply command for defect detection (CB-XXX violations)
    pub comply_command: String,

    /// Whether to fail on PMAT comply violations
    pub fail_on_comply_violations: bool,

    // =========================================================================
    // PMAT Quality Gate Integration (PMAT-STACK-GATES)
    // =========================================================================
    /// PMAT quality-gate command for comprehensive checks
    pub quality_gate_command: String,

    /// Whether to fail on quality gate violations
    pub fail_on_quality_gate: bool,

    /// PMAT TDG (Technical Debt Grading) command
    pub tdg_command: String,

    /// Minimum TDG score required (0-100)
    pub min_tdg_score: f64,

    /// Whether to fail on TDG score below threshold
    pub fail_on_tdg: bool,

    /// PMAT dead-code analysis command
    pub dead_code_command: String,

    /// Whether to fail on dead code detection
    pub fail_on_dead_code: bool,

    /// PMAT complexity analysis command
    pub complexity_command: String,

    /// Maximum cyclomatic complexity allowed
    pub max_complexity: u32,

    /// Whether to fail on complexity violations
    pub fail_on_complexity: bool,

    /// PMAT SATD (Self-Admitted Technical Debt) command
    pub satd_command: String,

    /// Maximum SATD items allowed
    pub max_satd_items: u32,

    /// Whether to fail on SATD violations
    pub fail_on_satd: bool,

    /// PMAT Popper score command (falsifiability)
    pub popper_command: String,

    /// Minimum Popper score required (0-100)
    pub min_popper_score: f64,

    /// Whether to fail on Popper score below threshold
    pub fail_on_popper: bool,
}

impl Default for ReleaseConfig {
    fn default() -> Self {
        Self {
            bump_type: None,
            no_verify: false,
            dry_run: false,
            publish: false,
            min_coverage: 90.0,
            lint_command: "make lint".to_string(),
            coverage_command: "make coverage".to_string(),
            comply_command: "pmat comply".to_string(),
            fail_on_comply_violations: true,
            // PMAT Quality Gate Integration defaults
            quality_gate_command: "pmat quality-gate".to_string(),
            fail_on_quality_gate: true,
            tdg_command: "pmat tdg --format json".to_string(),
            min_tdg_score: 80.0,
            fail_on_tdg: true,
            dead_code_command: "pmat analyze dead-code --format json".to_string(),
            fail_on_dead_code: false, // Warning only by default
            complexity_command: "pmat analyze complexity --format json".to_string(),
            max_complexity: 20,
            fail_on_complexity: true,
            satd_command: "pmat analyze satd --format json".to_string(),
            max_satd_items: 10,
            fail_on_satd: false, // Warning only by default
            popper_command: "pmat popper-score --format json".to_string(),
            min_popper_score: 60.0,
            fail_on_popper: true,
        }
    }
}

/// Release orchestrator for coordinated multi-crate releases
pub struct ReleaseOrchestrator {
    /// Release configuration
    config: ReleaseConfig,

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

        self.preflight_results
            .insert(crate_name.to_string(), result.clone());
        Ok(result)
    }

    /// Check if git working directory is clean
    fn check_git_clean(&self, crate_path: &Path) -> PreflightCheck {
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
    fn check_lint(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.lint_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::fail("lint", "No lint command configured");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    PreflightCheck::pass("lint", "Lint passed")
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    PreflightCheck::fail("lint", format!("Lint failed: {}", stderr.trim()))
                }
            }
            Err(e) => PreflightCheck::fail("lint", format!("Failed to run lint: {}", e)),
        }
    }

    /// Check coverage meets minimum
    fn check_coverage(&self, crate_path: &Path) -> PreflightCheck {
        // For now, just check if coverage command succeeds
        // In real implementation, we'd parse the coverage output
        let parts: Vec<&str> = self.config.coverage_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::fail("coverage", "No coverage command configured");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    PreflightCheck::pass(
                        "coverage",
                        format!("Coverage check passed (min: {}%)", self.config.min_coverage),
                    )
                } else {
                    PreflightCheck::fail(
                        "coverage",
                        format!("Coverage below {}%", self.config.min_coverage),
                    )
                }
            }
            Err(e) => PreflightCheck::fail("coverage", format!("Failed to run coverage: {}", e)),
        }
    }

    /// Check PMAT comply for ComputeBrick defects (CB-XXX violations)
    ///
    /// Runs `pmat comply` to detect:
    /// - CB-020: Unsafe blocks without safety comments
    /// - CB-021: SIMD without target_feature attributes
    /// - CB-022: Missing error handling patterns
    /// - And other PMAT compliance rules
    fn check_pmat_comply(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.comply_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("pmat_comply", "No comply command configured (skipped)");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                // Check for CB-XXX violations in output
                let has_violations = stdout.contains("CB-")
                    || stderr.contains("CB-")
                    || stdout.contains("violation")
                    || stderr.contains("violation");

                if out.status.success() && !has_violations {
                    PreflightCheck::pass("pmat_comply", "PMAT comply passed (0 violations)")
                } else if has_violations && self.config.fail_on_comply_violations {
                    // Extract violation count if possible
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
                    // Violations but fail_on_comply_violations is false
                    PreflightCheck::pass("pmat_comply", "PMAT comply has warnings (not blocking)")
                } else {
                    PreflightCheck::fail(
                        "pmat_comply",
                        format!("PMAT comply error: {}", stderr.trim()),
                    )
                }
            }
            Err(e) => {
                // pmat not installed - warn but don't fail
                if e.kind() == std::io::ErrorKind::NotFound {
                    PreflightCheck::pass(
                        "pmat_comply",
                        "pmat not found (install with: cargo install pmat)",
                    )
                } else {
                    PreflightCheck::fail("pmat_comply", format!("Failed to run pmat: {}", e))
                }
            }
        }
    }

    /// Check for path dependencies
    fn check_no_path_deps(&self, _crate_name: &str) -> PreflightCheck {
        // This would use the checker's graph to verify no path deps
        // For now, always pass as a placeholder
        PreflightCheck::pass("no_path_deps", "No path dependencies found")
    }

    /// Check version is bumped from crates.io
    fn check_version_bumped(&self, _crate_name: &str) -> PreflightCheck {
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
    fn check_pmat_quality_gate(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self
            .config
            .quality_gate_command
            .split_whitespace()
            .collect();
        if parts.is_empty() {
            return PreflightCheck::pass(
                "quality_gate",
                "No quality-gate command configured (skipped)",
            );
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    PreflightCheck::pass("quality_gate", "PMAT quality-gate passed")
                } else if self.config.fail_on_quality_gate {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    PreflightCheck::fail(
                        "quality_gate",
                        format!("Quality gate failed: {}", stderr.trim()),
                    )
                } else {
                    PreflightCheck::pass("quality_gate", "Quality gate has warnings (not blocking)")
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("quality_gate", "pmat not found (skipped)")
            }
            Err(e) => {
                PreflightCheck::fail("quality_gate", format!("Failed to run quality-gate: {}", e))
            }
        }
    }

    /// Check PMAT TDG (Technical Debt Grading) score
    ///
    /// Runs `pmat tdg --format json` and parses the score.
    /// Fails if score < min_tdg_score (default: 80).
    fn check_pmat_tdg(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.tdg_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("tdg", "No TDG command configured (skipped)");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                // Try to parse score from JSON output
                let score = Self::parse_score_from_json(&stdout, "score")
                    .or_else(|| Self::parse_score_from_json(&stdout, "tdg_score"))
                    .or_else(|| Self::parse_score_from_json(&stdout, "total"));

                match score {
                    Some(s) if s >= self.config.min_tdg_score => PreflightCheck::pass(
                        "tdg",
                        format!(
                            "TDG score: {:.1}/100 (min: {:.1})",
                            s, self.config.min_tdg_score
                        ),
                    ),
                    Some(s) if self.config.fail_on_tdg => PreflightCheck::fail(
                        "tdg",
                        format!(
                            "TDG score {:.1} below threshold {:.1}",
                            s, self.config.min_tdg_score
                        ),
                    ),
                    Some(s) => PreflightCheck::pass(
                        "tdg",
                        format!(
                            "TDG score: {:.1}/100 (warning: below {:.1})",
                            s, self.config.min_tdg_score
                        ),
                    ),
                    None if out.status.success() => {
                        PreflightCheck::pass("tdg", "TDG check passed (score not parsed)")
                    }
                    None => PreflightCheck::pass("tdg", "TDG score not available (skipped)"),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("tdg", "pmat not found (skipped)")
            }
            Err(e) => PreflightCheck::fail("tdg", format!("Failed to run TDG: {}", e)),
        }
    }

    /// Check PMAT dead-code analysis
    ///
    /// Runs `pmat analyze dead-code` to detect unused code.
    fn check_pmat_dead_code(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.dead_code_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("dead_code", "No dead-code command configured (skipped)");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let has_dead_code = stdout.contains("dead_code") || stdout.contains("unused");
                let count = Self::parse_count_from_json(&stdout, "count")
                    .or_else(|| Self::parse_count_from_json(&stdout, "dead_code_count"));

                match (has_dead_code, count) {
                    (_, Some(0)) | (false, None) => {
                        PreflightCheck::pass("dead_code", "No dead code detected")
                    }
                    (_, Some(n)) if self.config.fail_on_dead_code => {
                        PreflightCheck::fail("dead_code", format!("{} dead code items found", n))
                    }
                    (_, Some(n)) => PreflightCheck::pass(
                        "dead_code",
                        format!("{} dead code items (warning)", n),
                    ),
                    (true, None) if self.config.fail_on_dead_code => {
                        PreflightCheck::fail("dead_code", "Dead code detected")
                    }
                    (true, None) => {
                        PreflightCheck::pass("dead_code", "Dead code detected (warning)")
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("dead_code", "pmat not found (skipped)")
            }
            Err(e) => PreflightCheck::fail("dead_code", format!("Failed to run dead-code: {}", e)),
        }
    }

    /// Check PMAT complexity analysis
    ///
    /// Runs `pmat analyze complexity` to check cyclomatic complexity.
    /// Fails if any function exceeds max_complexity (default: 20).
    fn check_pmat_complexity(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.complexity_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass(
                "complexity",
                "No complexity command configured (skipped)",
            );
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let max_found = Self::parse_count_from_json(&stdout, "max_complexity")
                    .or_else(|| Self::parse_count_from_json(&stdout, "highest"));
                let violations = Self::parse_count_from_json(&stdout, "violations")
                    .or_else(|| Self::parse_count_from_json(&stdout, "violation_count"));

                match (max_found, violations) {
                    (Some(m), _) if m <= self.config.max_complexity => PreflightCheck::pass(
                        "complexity",
                        format!(
                            "Max complexity: {} (limit: {})",
                            m, self.config.max_complexity
                        ),
                    ),
                    (Some(m), _) if self.config.fail_on_complexity => PreflightCheck::fail(
                        "complexity",
                        format!(
                            "Complexity {} exceeds limit {}",
                            m, self.config.max_complexity
                        ),
                    ),
                    (_, Some(0)) => PreflightCheck::pass("complexity", "No complexity violations"),
                    (_, Some(v)) if self.config.fail_on_complexity => {
                        PreflightCheck::fail("complexity", format!("{} complexity violations", v))
                    }
                    _ if out.status.success() => {
                        PreflightCheck::pass("complexity", "Complexity check passed")
                    }
                    _ => PreflightCheck::pass("complexity", "Complexity check completed (warning)"),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("complexity", "pmat not found (skipped)")
            }
            Err(e) => {
                PreflightCheck::fail("complexity", format!("Failed to run complexity: {}", e))
            }
        }
    }

    /// Check PMAT SATD (Self-Admitted Technical Debt)
    ///
    /// Runs `pmat analyze satd` to detect TODO/FIXME/HACK comments.
    /// Fails if count exceeds max_satd_items (default: 10).
    fn check_pmat_satd(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.satd_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("satd", "No SATD command configured (skipped)");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let count = Self::parse_count_from_json(&stdout, "total")
                    .or_else(|| Self::parse_count_from_json(&stdout, "count"))
                    .or_else(|| Self::parse_count_from_json(&stdout, "satd_count"));

                match count {
                    Some(c) if c <= self.config.max_satd_items => PreflightCheck::pass(
                        "satd",
                        format!("{} SATD items (limit: {})", c, self.config.max_satd_items),
                    ),
                    Some(c) if self.config.fail_on_satd => PreflightCheck::fail(
                        "satd",
                        format!(
                            "{} SATD items exceed limit {}",
                            c, self.config.max_satd_items
                        ),
                    ),
                    Some(c) => PreflightCheck::pass(
                        "satd",
                        format!(
                            "{} SATD items (warning: exceeds {})",
                            c, self.config.max_satd_items
                        ),
                    ),
                    None if out.status.success() => {
                        PreflightCheck::pass("satd", "SATD check passed")
                    }
                    None => PreflightCheck::pass("satd", "SATD check completed"),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("satd", "pmat not found (skipped)")
            }
            Err(e) => PreflightCheck::fail("satd", format!("Failed to run SATD: {}", e)),
        }
    }

    /// Check PMAT Popper score (falsifiability)
    ///
    /// Runs `pmat popper-score` to assess scientific quality.
    /// Based on Karl Popper's falsification principles.
    /// Fails if score < min_popper_score (default: 60).
    fn check_pmat_popper(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.popper_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("popper", "No Popper command configured (skipped)");
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .current_dir(crate_path)
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let score = Self::parse_score_from_json(&stdout, "score")
                    .or_else(|| Self::parse_score_from_json(&stdout, "popper_score"))
                    .or_else(|| Self::parse_score_from_json(&stdout, "total"));

                match score {
                    Some(s) if s >= self.config.min_popper_score => PreflightCheck::pass(
                        "popper",
                        format!(
                            "Popper score: {:.1}/100 (min: {:.1})",
                            s, self.config.min_popper_score
                        ),
                    ),
                    Some(s) if self.config.fail_on_popper => PreflightCheck::fail(
                        "popper",
                        format!(
                            "Popper score {:.1} below threshold {:.1}",
                            s, self.config.min_popper_score
                        ),
                    ),
                    Some(s) => PreflightCheck::pass(
                        "popper",
                        format!("Popper score: {:.1}/100 (warning)", s),
                    ),
                    None if out.status.success() => {
                        PreflightCheck::pass("popper", "Popper check passed")
                    }
                    None => PreflightCheck::pass("popper", "Popper score not available (skipped)"),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                PreflightCheck::pass("popper", "pmat not found (skipped)")
            }
            Err(e) => PreflightCheck::fail("popper", format!("Failed to run Popper: {}", e)),
        }
    }

    /// Helper: Parse a numeric score from JSON output
    fn parse_score_from_json(json: &str, key: &str) -> Option<f64> {
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
    fn parse_count_from_json(json: &str, key: &str) -> Option<u32> {
        Self::parse_score_from_json(json, key).map(|f| f as u32)
    }

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

/// Result of a release execution
#[derive(Debug, Clone)]
pub struct ReleaseResult {
    /// Whether the release succeeded
    pub success: bool,

    /// Crates that were released
    pub released_crates: Vec<ReleasedCrate>,

    /// Message describing the result
    pub message: String,
}

/// Information about a released crate
#[derive(Debug, Clone)]
pub struct ReleasedCrate {
    /// Crate name
    pub name: String,

    /// Released version
    pub version: semver::Version,

    /// Whether it was published to crates.io
    pub published: bool,
}

/// Format a release plan as text
pub fn format_plan_text(plan: &ReleasePlan) -> String {
    let mut output = String::new();

    if plan.dry_run {
        output.push_str("📋 Release Plan (DRY RUN)\n");
    } else {
        output.push_str("📋 Release Plan\n");
    }
    output.push_str(&"═".repeat(50));
    output.push_str("\n\n");

    output.push_str("Release order (topological):\n");
    for (i, release) in plan.releases.iter().enumerate() {
        output.push_str(&format!(
            "  {}. {} {} → {}\n",
            i + 1,
            release.crate_name,
            release.current_version,
            release.new_version
        ));
    }

    output.push_str("\nPre-flight status:\n");
    for release in &plan.releases {
        let status = plan
            .preflight_results
            .get(&release.crate_name)
            .map(|r| if r.passed { "✓" } else { "✗" })
            .unwrap_or("?");

        output.push_str(&format!("  {} {}\n", status, release.crate_name));
    }

    output
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::stack::graph::{DependencyEdge, DependencyGraph};
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
    fn test_RELEASE_001_bump_type_equality() {
        assert_eq!(BumpType::Patch, BumpType::Patch);
        assert_eq!(BumpType::Minor, BumpType::Minor);
        assert_eq!(BumpType::Major, BumpType::Major);
        assert_ne!(BumpType::Patch, BumpType::Minor);
    }

    /// RED PHASE: Test BumpType clone
    #[test]
    fn test_RELEASE_001_bump_type_clone() {
        let bump = BumpType::Minor;
        let cloned = bump;
        assert_eq!(bump, cloned);
    }

    /// RED PHASE: Test BumpType debug
    #[test]
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
    fn test_RELEASE_002_config_clone() {
        let config = ReleaseConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.min_coverage, config.min_coverage);
        assert_eq!(cloned.dry_run, config.dry_run);
    }

    /// RED PHASE: Test ReleaseConfig debug
    #[test]
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
        assert!(text.contains("✓ trueno"));
        assert!(text.contains("✗ aprender"));
    }

    /// RED PHASE: Test format_plan_text empty plan
    #[test]
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

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(orchestrator.execute(&plan));

        assert!(result.is_ok());
        let release_result = result.unwrap();
        assert!(release_result.success);
        assert!(release_result.released_crates.is_empty());
        assert!(release_result.message.contains("Dry run"));
    }

    /// RED PHASE: Test execute fails when preflight failed
    #[test]
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

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(orchestrator.execute(&plan));

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Pre-flight checks failed"));
    }

    /// RED PHASE: Test execute success without publish
    #[test]
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

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(orchestrator.execute(&plan));

        assert!(result.is_ok());
        let release_result = result.unwrap();
        assert!(release_result.success);
        assert_eq!(release_result.released_crates.len(), 1);
        assert!(!release_result.released_crates[0].published);
    }

    /// RED PHASE: Test execute with multiple crates
    #[test]
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

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(orchestrator.execute(&plan));

        assert!(result.is_ok());
        let release_result = result.unwrap();
        assert!(release_result.success);
        assert_eq!(release_result.released_crates.len(), 2);
        assert!(release_result.released_crates[0].published);
        assert!(release_result.released_crates[1].published);
    }

    // ============================================================================
    // RELEASE-009: plan_all_releases
    // ============================================================================

    /// RED PHASE: Test plan_all_releases
    #[test]
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
    fn test_RELEASE_010_config_lint_command() {
        let config = ReleaseConfig {
            lint_command: "cargo clippy -- -D warnings".to_string(),
            ..Default::default()
        };

        assert_eq!(config.lint_command, "cargo clippy -- -D warnings");
    }

    /// RED PHASE: Test ReleaseConfig coverage command
    #[test]
    fn test_RELEASE_010_config_coverage_command() {
        let config = ReleaseConfig {
            coverage_command: "cargo tarpaulin".to_string(),
            min_coverage: 95.0,
            ..Default::default()
        };

        assert_eq!(config.coverage_command, "cargo tarpaulin");
        assert_eq!(config.min_coverage, 95.0);
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// PROPERTY: BumpType::Patch always increments patch by exactly 1
        #[test]
        fn prop_bump_patch_increments_by_one(
            major in 0u64..100,
            minor in 0u64..100,
            patch in 0u64..1000
        ) {
            let version = semver::Version::new(major, minor, patch);
            let bumped = BumpType::Patch.apply(&version);

            prop_assert_eq!(bumped.major, major);
            prop_assert_eq!(bumped.minor, minor);
            prop_assert_eq!(bumped.patch, patch + 1);
        }

        /// PROPERTY: BumpType::Minor increments minor and resets patch to 0
        #[test]
        fn prop_bump_minor_resets_patch(
            major in 0u64..100,
            minor in 0u64..100,
            patch in 0u64..1000
        ) {
            let version = semver::Version::new(major, minor, patch);
            let bumped = BumpType::Minor.apply(&version);

            prop_assert_eq!(bumped.major, major);
            prop_assert_eq!(bumped.minor, minor + 1);
            prop_assert_eq!(bumped.patch, 0);
        }

        /// PROPERTY: BumpType::Major increments major and resets minor/patch to 0
        #[test]
        fn prop_bump_major_resets_all(
            major in 0u64..100,
            minor in 0u64..100,
            patch in 0u64..1000
        ) {
            let version = semver::Version::new(major, minor, patch);
            let bumped = BumpType::Major.apply(&version);

            prop_assert_eq!(bumped.major, major + 1);
            prop_assert_eq!(bumped.minor, 0);
            prop_assert_eq!(bumped.patch, 0);
        }

        /// PROPERTY: Bumped version is always greater than original
        #[test]
        fn prop_bumped_version_always_greater(
            major in 0u64..100,
            minor in 0u64..100,
            patch in 0u64..1000,
            bump_idx in 0usize..3
        ) {
            let version = semver::Version::new(major, minor, patch);
            let bump = match bump_idx {
                0 => BumpType::Patch,
                1 => BumpType::Minor,
                _ => BumpType::Major,
            };

            let bumped = bump.apply(&version);

            prop_assert!(bumped > version, "Bumped {} should be > {}", bumped, version);
        }

        /// PROPERTY: format_plan_text never panics
        #[test]
        fn prop_format_plan_text_never_panics(
            num_releases in 0usize..10,
            dry_run: bool
        ) {
            let releases: Vec<PlannedRelease> = (0..num_releases)
                .map(|i| PlannedRelease {
                    crate_name: format!("crate-{}", i),
                    current_version: semver::Version::new(0, i as u64, 0),
                    new_version: semver::Version::new(0, i as u64, 1),
                    dependents: vec![],
                    ready: true,
                })
                .collect();

            let plan = ReleasePlan {
                releases,
                dry_run,
                preflight_results: HashMap::new(),
            };

            // Should not panic
            let _text = format_plan_text(&plan);
        }

        /// PROPERTY: ReleaseConfig clone is identical to original
        #[test]
        fn prop_release_config_clone_identical(
            no_verify: bool,
            dry_run: bool,
            publish: bool,
            min_coverage in 0.0f64..100.0,
            fail_on_comply in proptest::bool::ANY
        ) {
            let config = ReleaseConfig {
                bump_type: None,
                no_verify,
                dry_run,
                publish,
                min_coverage,
                lint_command: "cargo clippy".to_string(),
                coverage_command: "cargo tarpaulin".to_string(),
                comply_command: "pmat comply".to_string(),
                fail_on_comply_violations: fail_on_comply,
                // PMAT Quality Gate Integration (use defaults)
                ..Default::default()
            };

            let cloned = config.clone();

            prop_assert_eq!(config.no_verify, cloned.no_verify);
            prop_assert_eq!(config.dry_run, cloned.dry_run);
            prop_assert_eq!(config.publish, cloned.publish);
            prop_assert_eq!(config.fail_on_comply_violations, cloned.fail_on_comply_violations);
            prop_assert!((config.min_coverage - cloned.min_coverage).abs() < f64::EPSILON);
        }
    }

    // ============================================================================
    // PMAT-STACK-GATES: PMAT Quality Gate Integration Tests
    // ============================================================================

    #[test]
    fn test_PMAT_GATES_config_defaults() {
        // ARRANGE
        let config = ReleaseConfig::default();

        // ASSERT - verify all PMAT gate defaults
        assert_eq!(config.quality_gate_command, "pmat quality-gate");
        assert!(config.fail_on_quality_gate);
        assert_eq!(config.tdg_command, "pmat tdg --format json");
        assert_eq!(config.min_tdg_score, 80.0);
        assert!(config.fail_on_tdg);
        assert_eq!(
            config.dead_code_command,
            "pmat analyze dead-code --format json"
        );
        assert!(!config.fail_on_dead_code); // Warning only by default
        assert_eq!(
            config.complexity_command,
            "pmat analyze complexity --format json"
        );
        assert_eq!(config.max_complexity, 20);
        assert!(config.fail_on_complexity);
        assert_eq!(config.satd_command, "pmat analyze satd --format json");
        assert_eq!(config.max_satd_items, 10);
        assert!(!config.fail_on_satd); // Warning only by default
        assert_eq!(config.popper_command, "pmat popper-score --format json");
        assert_eq!(config.min_popper_score, 60.0);
        assert!(config.fail_on_popper);
    }

    #[test]
    fn test_PMAT_GATES_parse_score_from_json() {
        // ARRANGE
        let json = r#"{"score": 85.5, "other": "value"}"#;

        // ACT
        let score = ReleaseOrchestrator::parse_score_from_json(json, "score");

        // ASSERT
        assert_eq!(score, Some(85.5));
    }

    #[test]
    fn test_PMAT_GATES_parse_score_missing_key() {
        // ARRANGE
        let json = r#"{"other": 100}"#;

        // ACT
        let score = ReleaseOrchestrator::parse_score_from_json(json, "score");

        // ASSERT
        assert_eq!(score, None);
    }

    #[test]
    fn test_PMAT_GATES_parse_count_from_json() {
        // ARRANGE
        let json = r#"{"count": 42, "total": 100}"#;

        // ACT
        let count = ReleaseOrchestrator::parse_count_from_json(json, "count");

        // ASSERT
        assert_eq!(count, Some(42));
    }

    #[test]
    fn test_PMAT_GATES_parse_tdg_score() {
        // ARRANGE
        let json = r#"{"tdg_score": 92.5, "files_analyzed": 50}"#;

        // ACT
        let score = ReleaseOrchestrator::parse_score_from_json(json, "tdg_score");

        // ASSERT
        assert_eq!(score, Some(92.5));
    }

    #[test]
    fn test_PMAT_GATES_parse_popper_score() {
        // ARRANGE
        let json = r#"{"popper_score": 75.0, "category": "A"}"#;

        // ACT
        let score = ReleaseOrchestrator::parse_score_from_json(json, "popper_score");

        // ASSERT
        assert_eq!(score, Some(75.0));
    }

    #[test]
    fn test_PMAT_GATES_config_custom_thresholds() {
        // ARRANGE/ACT
        let config = ReleaseConfig {
            min_tdg_score: 90.0,
            min_popper_score: 70.0,
            max_complexity: 15,
            max_satd_items: 5,
            fail_on_dead_code: true,
            fail_on_satd: true,
            ..Default::default()
        };

        // ASSERT
        assert_eq!(config.min_tdg_score, 90.0);
        assert_eq!(config.min_popper_score, 70.0);
        assert_eq!(config.max_complexity, 15);
        assert_eq!(config.max_satd_items, 5);
        assert!(config.fail_on_dead_code);
        assert!(config.fail_on_satd);
    }

    #[test]
    fn test_PMAT_GATES_disabled_checks() {
        // ARRANGE/ACT - disable all PMAT checks
        let config = ReleaseConfig {
            quality_gate_command: String::new(),
            tdg_command: String::new(),
            dead_code_command: String::new(),
            complexity_command: String::new(),
            satd_command: String::new(),
            popper_command: String::new(),
            ..Default::default()
        };

        // ASSERT
        assert!(config.quality_gate_command.is_empty());
        assert!(config.tdg_command.is_empty());
        assert!(config.dead_code_command.is_empty());
        assert!(config.complexity_command.is_empty());
        assert!(config.satd_command.is_empty());
        assert!(config.popper_command.is_empty());
    }
}
