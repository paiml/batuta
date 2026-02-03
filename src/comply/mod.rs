//! Stack Compliance Engine - Cross-project consistency enforcement
//!
//! Ensures consistent patterns across the Sovereign AI Stack:
//! - Makefile target consistency
//! - Cargo.toml parity
//! - CI workflow alignment
//! - Code duplication detection (MinHash+LSH)
//!
//! # Toyota Production System Principles
//!
//! - **Heijunka**: Standardized targets across all projects
//! - **Poka-Yoke**: Configuration validation prevents drift
//! - **Jidoka**: Stop-on-error for critical violations
//! - **Kaizen**: Continuous improvement via compliance tracking

// Allow unused for public API items not yet consumed
#![allow(dead_code)]

pub mod config;
pub mod report;
pub mod rule;
pub mod rules;

pub use config::ComplyConfig;
#[allow(unused_imports)]
pub use config::ProjectOverride;
pub use report::{ComplyReport, ComplyReportFormat};
#[allow(unused_imports)]
pub use rule::{FixResult, RuleResult};
pub use rule::StackComplianceRule;

use crate::stack::PAIML_CRATES;
use std::path::{Path, PathBuf};

/// Stack Compliance Engine
///
/// Orchestrates compliance checks across all PAIML stack projects.
#[derive(Debug)]
pub struct StackComplyEngine {
    /// Compliance configuration
    config: ComplyConfig,
    /// Registered compliance rules
    rules: Vec<Box<dyn StackComplianceRule>>,
    /// Project discovery cache
    discovered_projects: Vec<ProjectInfo>,
}

/// Information about a discovered project
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    /// Project name (crate name)
    pub name: String,
    /// Path to project root
    pub path: PathBuf,
    /// Whether it's a PAIML stack crate
    pub is_paiml_crate: bool,
}

impl StackComplyEngine {
    /// Create a new compliance engine with default rules
    pub fn new(config: ComplyConfig) -> Self {
        let mut engine = Self {
            config,
            rules: Vec::new(),
            discovered_projects: Vec::new(),
        };

        // Register default rules
        engine.register_rule(Box::new(rules::MakefileRule::new()));
        engine.register_rule(Box::new(rules::CargoTomlRule::new()));
        engine.register_rule(Box::new(rules::CiWorkflowRule::new()));
        engine.register_rule(Box::new(rules::DuplicationRule::new()));

        engine
    }

    /// Create engine with default configuration
    pub fn default_for_workspace(workspace: &Path) -> Self {
        Self::new(ComplyConfig::default_for_workspace(workspace))
    }

    /// Register a custom compliance rule
    pub fn register_rule(&mut self, rule: Box<dyn StackComplianceRule>) {
        self.rules.push(rule);
    }

    /// Discover projects in the workspace
    pub fn discover_projects(&mut self, workspace: &Path) -> anyhow::Result<&[ProjectInfo]> {
        self.discovered_projects.clear();

        // Walk workspace looking for Cargo.toml files
        for entry in walkdir::WalkDir::new(workspace)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.file_name() == Some(std::ffi::OsStr::new("Cargo.toml")) {
                if let Some(project) = self.parse_project(path)? {
                    self.discovered_projects.push(project);
                }
            }
        }

        Ok(&self.discovered_projects)
    }

    /// Parse a project from its Cargo.toml
    fn parse_project(&self, cargo_toml: &Path) -> anyhow::Result<Option<ProjectInfo>> {
        let content = std::fs::read_to_string(cargo_toml)?;
        let toml: toml::Value = toml::from_str(&content)?;

        let name = toml
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .map(String::from);

        match name {
            Some(name) => {
                let path = cargo_toml.parent().unwrap_or(Path::new(".")).to_path_buf();
                let is_paiml_crate = PAIML_CRATES.contains(&name.as_str());

                Ok(Some(ProjectInfo {
                    name,
                    path,
                    is_paiml_crate,
                }))
            }
            None => Ok(None),
        }
    }

    /// Run all compliance checks
    pub fn check_all(&self) -> ComplyReport {
        let mut report = ComplyReport::new();

        for project in &self.discovered_projects {
            // Skip non-PAIML crates unless explicitly included
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            for rule in &self.rules {
                // Check if rule is enabled
                if !self.is_rule_enabled(rule.id()) {
                    continue;
                }

                // Check for project-specific override
                if self.has_rule_exemption(&project.name, rule.id()) {
                    report.add_exemption(&project.name, rule.id());
                    continue;
                }

                match rule.check(&project.path) {
                    Ok(result) => {
                        report.add_result(&project.name, rule.id(), result);
                    }
                    Err(e) => {
                        report.add_error(&project.name, rule.id(), e.to_string());
                    }
                }
            }
        }

        report.finalize();
        report
    }

    /// Run a specific rule
    pub fn check_rule(&self, rule_id: &str) -> ComplyReport {
        let mut report = ComplyReport::new();

        let rule = match self.rules.iter().find(|r| r.id() == rule_id) {
            Some(r) => r,
            None => {
                report.add_global_error(format!("Unknown rule: {}", rule_id));
                return report;
            }
        };

        for project in &self.discovered_projects {
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            if self.has_rule_exemption(&project.name, rule_id) {
                report.add_exemption(&project.name, rule_id);
                continue;
            }

            match rule.check(&project.path) {
                Ok(result) => {
                    report.add_result(&project.name, rule_id, result);
                }
                Err(e) => {
                    report.add_error(&project.name, rule_id, e.to_string());
                }
            }
        }

        report.finalize();
        report
    }

    /// Attempt to fix violations
    pub fn fix_all(&self, dry_run: bool) -> ComplyReport {
        let mut report = ComplyReport::new();

        for project in &self.discovered_projects {
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            for rule in &self.rules {
                if !self.is_rule_enabled(rule.id()) || !rule.can_fix() {
                    continue;
                }

                if self.has_rule_exemption(&project.name, rule.id()) {
                    continue;
                }

                // First check if there are violations to fix
                let check_result = match rule.check(&project.path) {
                    Ok(r) => r,
                    Err(e) => {
                        report.add_error(&project.name, rule.id(), e.to_string());
                        continue;
                    }
                };

                if check_result.passed {
                    report.add_result(&project.name, rule.id(), check_result);
                    continue;
                }

                // Attempt fix
                if dry_run {
                    report.add_dry_run_fix(&project.name, rule.id(), &check_result.violations);
                } else {
                    match rule.fix(&project.path) {
                        Ok(fix_result) => {
                            report.add_fix_result(&project.name, rule.id(), fix_result);
                        }
                        Err(e) => {
                            report.add_error(&project.name, rule.id(), e.to_string());
                        }
                    }
                }
            }
        }

        report.finalize();
        report
    }

    /// Check if a rule is enabled
    fn is_rule_enabled(&self, rule_id: &str) -> bool {
        if self.config.enabled_rules.is_empty() {
            // All rules enabled by default
            !self.config.disabled_rules.contains(&rule_id.to_string())
        } else {
            self.config.enabled_rules.contains(&rule_id.to_string())
        }
    }

    /// Check if a project has an exemption for a rule
    fn has_rule_exemption(&self, project_name: &str, rule_id: &str) -> bool {
        self.config
            .project_overrides
            .get(project_name)
            .map(|o| o.exempt_rules.contains(&rule_id.to_string()))
            .unwrap_or(false)
    }

    /// Get list of available rules
    pub fn available_rules(&self) -> Vec<(&str, &str)> {
        self.rules
            .iter()
            .map(|r| (r.id(), r.description()))
            .collect()
    }

    /// Get list of discovered projects
    pub fn projects(&self) -> &[ProjectInfo] {
        &self.discovered_projects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comply_engine_creation() {
        let config = ComplyConfig::default();
        let engine = StackComplyEngine::new(config);
        assert!(!engine.rules.is_empty());
    }

    #[test]
    fn test_available_rules() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let rules = engine.available_rules();
        assert!(rules.iter().any(|(id, _)| *id == "makefile-targets"));
        assert!(rules.iter().any(|(id, _)| *id == "cargo-toml-consistency"));
        assert!(rules.iter().any(|(id, _)| *id == "ci-workflow-parity"));
        assert!(rules.iter().any(|(id, _)| *id == "code-duplication"));
    }

    #[test]
    fn test_rule_enabled_check() {
        let mut config = ComplyConfig::default();
        config.disabled_rules.push("makefile-targets".to_string());
        let engine = StackComplyEngine::new(config);
        assert!(!engine.is_rule_enabled("makefile-targets"));
        assert!(engine.is_rule_enabled("cargo-toml-consistency"));
    }

    #[test]
    fn test_project_exemption() {
        let mut config = ComplyConfig::default();
        let mut override_config = ProjectOverride::default();
        override_config.exempt_rules.push("makefile-targets".to_string());
        config
            .project_overrides
            .insert("test-project".to_string(), override_config);

        let engine = StackComplyEngine::new(config);
        assert!(engine.has_rule_exemption("test-project", "makefile-targets"));
        assert!(!engine.has_rule_exemption("test-project", "cargo-toml-consistency"));
        assert!(!engine.has_rule_exemption("other-project", "makefile-targets"));
    }

    #[test]
    fn test_default_for_workspace() {
        let engine = StackComplyEngine::default_for_workspace(Path::new("."));
        assert!(!engine.rules.is_empty());
    }

    #[test]
    fn test_projects_empty_initially() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        assert!(engine.projects().is_empty());
    }

    #[test]
    fn test_enabled_rules_explicit() {
        let mut config = ComplyConfig::default();
        config.enabled_rules.push("makefile-targets".to_string());
        let engine = StackComplyEngine::new(config);
        assert!(engine.is_rule_enabled("makefile-targets"));
        assert!(!engine.is_rule_enabled("cargo-toml-consistency"));
    }

    #[test]
    fn test_project_info_fields() {
        let info = ProjectInfo {
            name: "test-project".to_string(),
            path: PathBuf::from("/path/to/project"),
            is_paiml_crate: true,
        };
        assert_eq!(info.name, "test-project");
        assert_eq!(info.path, PathBuf::from("/path/to/project"));
        assert!(info.is_paiml_crate);
    }

    #[test]
    fn test_check_rule_unknown() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.check_rule("nonexistent-rule");
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_check_all_empty_projects() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.check_all();
        // With no discovered projects, should return empty report
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_empty_projects() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.fix_all(true);
        assert_eq!(report.summary.total_projects, 0);
    }
}
