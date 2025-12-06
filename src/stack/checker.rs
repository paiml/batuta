#![allow(dead_code)]
//! Stack Health Checker
//!
//! Implements the `batuta stack check` command functionality.
//! Analyzes the PAIML stack for dependency issues, version conflicts,
//! and path dependencies that should be crates.io versions.

use crate::stack::crates_io::{CratesIoClient, MockCratesIoClient};
use crate::stack::graph::DependencyGraph;
use crate::stack::types::*;
use anyhow::Result;
use std::path::Path;

/// Stack health checker
pub struct StackChecker {
    /// Dependency graph
    graph: DependencyGraph,

    /// Whether to verify against crates.io
    verify_published: bool,

    /// Strict mode (fail on warnings)
    strict: bool,
}

impl StackChecker {
    /// Create a new stack checker from a workspace path
    #[cfg(feature = "native")]
    pub fn from_workspace(workspace_path: &Path) -> Result<Self> {
        let graph = DependencyGraph::from_workspace(workspace_path)?;
        Ok(Self {
            graph,
            verify_published: false,
            strict: false,
        })
    }

    /// Create a stack checker with an existing graph (for testing)
    pub fn with_graph(graph: DependencyGraph) -> Self {
        Self {
            graph,
            verify_published: false,
            strict: false,
        }
    }

    /// Enable crates.io verification
    pub fn verify_published(mut self, verify: bool) -> Self {
        self.verify_published = verify;
        self
    }

    /// Enable strict mode
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Run health check with mock crates.io client (for testing)
    pub fn check_with_mock(&mut self, mock: &MockCratesIoClient) -> Result<StackHealthReport> {
        self.run_checks(|name| mock.get_latest_version(name).ok())
    }

    /// Run health check with real crates.io client
    #[cfg(feature = "native")]
    pub async fn check(&mut self, client: &mut CratesIoClient) -> Result<StackHealthReport> {
        // First, fetch all crates.io versions
        let mut crates_io_versions = std::collections::HashMap::new();

        if self.verify_published {
            for crate_info in self.graph.all_crates() {
                if let Ok(version) = client.get_latest_version(&crate_info.name).await {
                    crates_io_versions.insert(crate_info.name.clone(), version);
                }
            }
        }

        self.run_checks(|name| crates_io_versions.get(name).cloned())
    }

    /// Internal check implementation
    fn run_checks<F>(&mut self, get_crates_io_version: F) -> Result<StackHealthReport>
    where
        F: Fn(&str) -> Option<semver::Version>,
    {
        // Check for cycles first
        // Note: Cycle detection is done at the graph level
        // If cycles exist, topological_order() will fail

        // Find path dependencies
        let path_deps = self.graph.find_path_dependencies();

        // Detect version conflicts
        let conflicts = self.graph.detect_conflicts();

        // Update crate statuses and issues
        let mut crates: Vec<CrateInfo> = Vec::new();

        for crate_info in self.graph.all_crates() {
            let mut info = crate_info.clone();

            // Update crates.io version
            info.crates_io_version = get_crates_io_version(&info.name);

            // Check for path dependencies
            for path_dep in &path_deps {
                if path_dep.crate_name == info.name {
                    let suggestion = info
                        .crates_io_version
                        .as_ref()
                        .map(|v| format!("{} = \"{}\"", path_dep.dependency, v));

                    let mut issue = CrateIssue::new(
                        IssueSeverity::Error,
                        IssueType::PathDependency,
                        format!(
                            "Path dependency '{}' should be a crates.io version",
                            path_dep.dependency
                        ),
                    );

                    if let Some(sug) = suggestion {
                        issue = issue.with_suggestion(sug);
                    }

                    info.issues.push(issue);
                }
            }

            // Check for version conflicts
            for conflict in &conflicts {
                let is_involved = conflict.usages.iter().any(|u| u.crate_name == info.name);

                if is_involved {
                    info.issues.push(CrateIssue::new(
                        IssueSeverity::Warning,
                        IssueType::VersionConflict,
                        format!(
                            "Version conflict for '{}': different versions required across stack",
                            conflict.dependency
                        ),
                    ));
                }
            }

            // Check if not published
            if self.verify_published && info.crates_io_version.is_none() {
                info.issues.push(CrateIssue::new(
                    IssueSeverity::Info,
                    IssueType::NotPublished,
                    format!("Crate '{}' is not published to crates.io", info.name),
                ));
            }

            // Check if version is behind
            if let Some(ref remote) = info.crates_io_version {
                if info.local_version < *remote {
                    info.issues.push(CrateIssue::new(
                        IssueSeverity::Warning,
                        IssueType::VersionBehind,
                        format!(
                            "Local version {} is behind crates.io version {}",
                            info.local_version, remote
                        ),
                    ));
                }
            }

            // Determine status based on issues
            info.status = Self::determine_status(&info.issues, self.strict);

            crates.push(info);
        }

        // Sort crates by name for consistent output
        crates.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(StackHealthReport::new(crates, conflicts))
    }

    /// Determine crate status based on issues
    fn determine_status(issues: &[CrateIssue], strict: bool) -> CrateStatus {
        let has_errors = issues.iter().any(|i| i.severity == IssueSeverity::Error);
        let has_warnings = issues.iter().any(|i| i.severity == IssueSeverity::Warning);

        if has_errors {
            CrateStatus::Error
        } else if has_warnings {
            if strict {
                CrateStatus::Error
            } else {
                CrateStatus::Warning
            }
        } else {
            CrateStatus::Healthy
        }
    }

    /// Get the release order for a specific crate
    pub fn release_order_for(&self, crate_name: &str) -> Result<Vec<String>> {
        self.graph.release_order_for(crate_name)
    }

    /// Get topological order for all crates
    pub fn topological_order(&self) -> Result<Vec<String>> {
        self.graph.topological_order()
    }

    /// Get number of crates in the stack
    pub fn crate_count(&self) -> usize {
        self.graph.crate_count()
    }
}

/// Format a health report as text
pub fn format_report_text(report: &StackHealthReport) -> String {
    let mut output = String::new();

    output.push_str("🔍 PAIML Stack Health Check\n");
    output.push_str(&"═".repeat(60));
    output.push_str("\n\n");

    for crate_info in &report.crates {
        let status_icon = match crate_info.status {
            CrateStatus::Healthy => "✅",
            CrateStatus::Warning => "⚠️ ",
            CrateStatus::Error => "❌",
            CrateStatus::Unknown => "❓",
        };

        let crates_io_str = match &crate_info.crates_io_version {
            Some(v) => format!("(crates.io: {})", v),
            None => "(not published)".to_string(),
        };

        output.push_str(&format!(
            "{} {} v{} {}\n",
            status_icon, crate_info.name, crate_info.local_version, crates_io_str
        ));

        for issue in &crate_info.issues {
            let severity_prefix = match issue.severity {
                IssueSeverity::Error => "  ✗",
                IssueSeverity::Warning => "  ⚠",
                IssueSeverity::Info => "  ℹ",
            };

            output.push_str(&format!("{}  {}\n", severity_prefix, issue.message));

            if let Some(ref suggestion) = issue.suggestion {
                output.push_str(&format!("      → {}\n", suggestion));
            }
        }

        output.push('\n');
    }

    // Conflicts section
    if !report.conflicts.is_empty() {
        output.push_str("Version Conflicts:\n");
        output.push_str(&"─".repeat(40));
        output.push('\n');

        for conflict in &report.conflicts {
            output.push_str(&format!("  {} conflict:\n", conflict.dependency));
            for usage in &conflict.usages {
                output.push_str(&format!(
                    "    - {} requires {}\n",
                    usage.crate_name, usage.version_req
                ));
            }
            if let Some(ref rec) = conflict.recommendation {
                output.push_str(&format!("    Recommendation: {}\n", rec));
            }
            output.push('\n');
        }
    }

    // Summary
    output.push_str(&"─".repeat(60));
    output.push('\n');
    output.push_str("Summary:\n");
    output.push_str(&format!(
        "  Total crates: {}\n",
        report.summary.total_crates
    ));
    output.push_str(&format!("  Healthy: {}\n", report.summary.healthy_count));
    output.push_str(&format!("  Warnings: {}\n", report.summary.warning_count));
    output.push_str(&format!("  Errors: {}\n", report.summary.error_count));

    if report.summary.path_dependency_count > 0 {
        output.push_str(&format!(
            "  Path dependencies: {}\n",
            report.summary.path_dependency_count
        ));
    }

    output
}

/// Format a health report as Markdown
pub fn format_report_markdown(report: &StackHealthReport) -> String {
    let mut output = String::new();

    output.push_str("# PAIML Stack Health Report\n\n");

    output.push_str("## Crates\n\n");
    output.push_str("| Status | Crate | Version | Crates.io |\n");
    output.push_str("|--------|-------|---------|----------|\n");

    for crate_info in &report.crates {
        let status_icon = match crate_info.status {
            CrateStatus::Healthy => "✅",
            CrateStatus::Warning => "⚠️",
            CrateStatus::Error => "❌",
            CrateStatus::Unknown => "❓",
        };

        let crates_io_str = match &crate_info.crates_io_version {
            Some(v) => v.to_string(),
            None => "not published".to_string(),
        };

        output.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            status_icon, crate_info.name, crate_info.local_version, crates_io_str
        ));

        // Add issues as sub-items
        for issue in &crate_info.issues {
            let icon = match issue.severity {
                IssueSeverity::Error => "❌",
                IssueSeverity::Warning => "⚠️",
                IssueSeverity::Info => "ℹ️",
            };
            output.push_str(&format!("| | | {} {} | |\n", icon, issue.message));
        }
    }

    output.push_str("\n## Summary\n\n");
    output.push_str(&format!("- **Total crates**: {}\n", report.summary.total_crates));
    output.push_str(&format!("- **Healthy**: {}\n", report.summary.healthy_count));
    output.push_str(&format!("- **Warnings**: {}\n", report.summary.warning_count));
    output.push_str(&format!("- **Errors**: {}\n", report.summary.error_count));

    if report.is_healthy() {
        output.push_str("\n✅ **All crates are healthy**\n");
    } else {
        output.push_str("\n⚠️ **Some crates need attention**\n");
    }

    output
}

/// Format a health report as JSON
pub fn format_report_json(report: &StackHealthReport) -> Result<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| anyhow::anyhow!("JSON serialization error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stack::graph::DependencyEdge;
    use std::path::PathBuf;

    fn create_test_graph() -> DependencyGraph {
        let mut graph = DependencyGraph::new();

        // trueno - healthy crate
        graph.add_crate(CrateInfo::new(
            "trueno",
            semver::Version::new(1, 2, 0),
            PathBuf::from("trueno/Cargo.toml"),
        ));

        // aprender - depends on trueno
        let mut aprender = CrateInfo::new(
            "aprender",
            semver::Version::new(0, 8, 1),
            PathBuf::from("aprender/Cargo.toml"),
        );
        aprender
            .paiml_dependencies
            .push(DependencyInfo::new("trueno", "^1.0"));
        graph.add_crate(aprender);

        // entrenar - has PATH DEPENDENCY (the bug!)
        let mut entrenar = CrateInfo::new(
            "entrenar",
            semver::Version::new(0, 2, 2),
            PathBuf::from("entrenar/Cargo.toml"),
        );
        entrenar
            .paiml_dependencies
            .push(DependencyInfo::new("aprender", "^0.8"));
        entrenar.paiml_dependencies.push(DependencyInfo::path(
            "alimentar",
            PathBuf::from("../alimentar"),
        ));
        graph.add_crate(entrenar);

        // alimentar
        graph.add_crate(CrateInfo::new(
            "alimentar",
            semver::Version::new(0, 3, 0),
            PathBuf::from("alimentar/Cargo.toml"),
        ));

        // Add edges
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

        graph.add_dependency(
            "entrenar",
            "alimentar",
            DependencyEdge {
                version_req: String::new(),
                is_path: true,
                kind: DependencyKind::Normal,
            },
        );

        graph.add_dependency(
            "alimentar",
            "trueno",
            DependencyEdge {
                version_req: "^1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        graph
    }

    fn create_mock_client() -> MockCratesIoClient {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("trueno", "1.2.0")
            .add_crate("aprender", "0.8.1")
            .add_crate("alimentar", "0.3.0")
            .add_crate("entrenar", "0.2.2");
        mock
    }

    #[test]
    fn test_checker_creation() {
        let graph = create_test_graph();
        let checker = StackChecker::with_graph(graph);
        assert_eq!(checker.crate_count(), 4);
    }

    #[test]
    fn test_checker_finds_path_dependencies() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();

        // Find entrenar in the report
        let entrenar = report.crates.iter().find(|c| c.name == "entrenar").unwrap();

        // Should have path dependency issue
        assert!(entrenar
            .issues
            .iter()
            .any(|i| i.issue_type == IssueType::PathDependency));
        assert_eq!(entrenar.status, CrateStatus::Error);
    }

    #[test]
    fn test_checker_healthy_crates() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();

        // trueno should be healthy
        let trueno = report.crates.iter().find(|c| c.name == "trueno").unwrap();
        assert_eq!(trueno.status, CrateStatus::Healthy);
        assert!(trueno.issues.is_empty());
    }

    #[test]
    fn test_checker_crates_io_versions() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph).verify_published(true);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();

        // All crates should have crates.io versions
        for crate_info in &report.crates {
            assert!(crate_info.crates_io_version.is_some());
        }
    }

    #[test]
    fn test_checker_strict_mode() {
        let mut graph = DependencyGraph::new();

        // Create crate with warning (version behind)
        let crate_info = CrateInfo::new(
            "test-crate",
            semver::Version::new(0, 9, 0), // Behind crates.io
            PathBuf::new(),
        );
        graph.add_crate(crate_info);

        let mut mock = MockCratesIoClient::new();
        mock.add_crate("test-crate", "1.0.0"); // Ahead

        // Non-strict mode
        let mut checker = StackChecker::with_graph(graph.clone()).verify_published(true);
        let report = checker.check_with_mock(&mock).unwrap();
        let crate_info = report
            .crates
            .iter()
            .find(|c| c.name == "test-crate")
            .unwrap();
        assert_eq!(crate_info.status, CrateStatus::Warning);

        // Strict mode
        let mut checker = StackChecker::with_graph(graph)
            .verify_published(true)
            .strict(true);
        let report = checker.check_with_mock(&mock).unwrap();
        let crate_info = report
            .crates
            .iter()
            .find(|c| c.name == "test-crate")
            .unwrap();
        assert_eq!(crate_info.status, CrateStatus::Error); // Warning becomes error in strict mode
    }

    #[test]
    fn test_checker_release_order() {
        let graph = create_test_graph();
        let checker = StackChecker::with_graph(graph);

        let order = checker.release_order_for("entrenar").unwrap();

        // trueno should be first
        assert_eq!(order[0], "trueno");

        // entrenar should be last
        assert_eq!(order.last().unwrap(), "entrenar");

        // All dependencies should be included
        assert!(order.contains(&"aprender".to_string()));
        assert!(order.contains(&"alimentar".to_string()));
    }

    #[test]
    fn test_health_report_is_healthy() {
        let mut graph = DependencyGraph::new();

        let mut healthy_crate =
            CrateInfo::new("healthy", semver::Version::new(1, 0, 0), PathBuf::new());
        healthy_crate.status = CrateStatus::Healthy;
        graph.add_crate(healthy_crate);

        let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);

        assert!(report.is_healthy());
    }

    #[test]
    fn test_health_report_not_healthy_with_errors() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();

        // Should not be healthy due to path dependency in entrenar
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_format_report_text() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();
        let text = format_report_text(&report);

        // Should contain key information
        assert!(text.contains("PAIML Stack Health Check"));
        assert!(text.contains("trueno"));
        assert!(text.contains("entrenar"));
        assert!(text.contains("Summary:"));
    }

    #[test]
    fn test_format_report_json() {
        let graph = create_test_graph();
        let mut checker = StackChecker::with_graph(graph);
        let mock = create_mock_client();

        let report = checker.check_with_mock(&mock).unwrap();
        let json = format_report_json(&report).unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("crates").is_some());
        assert!(parsed.get("summary").is_some());
    }

    #[test]
    fn test_version_conflict_detection() {
        let mut graph = DependencyGraph::new();

        // Create crates with conflicting arrow versions
        let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new());
        crate_a
            .external_dependencies
            .push(DependencyInfo::new("arrow", "54.0"));
        graph.add_crate(crate_a);

        let mut crate_b = CrateInfo::new("b", semver::Version::new(1, 0, 0), PathBuf::new());
        crate_b
            .external_dependencies
            .push(DependencyInfo::new("arrow", "53.0"));
        graph.add_crate(crate_b);

        let mut checker = StackChecker::with_graph(graph);
        let mock = MockCratesIoClient::new();

        let report = checker.check_with_mock(&mock).unwrap();

        // Should detect conflict
        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(report.conflicts[0].dependency, "arrow");
    }

    #[test]
    fn test_determine_status() {
        // No issues = healthy
        assert_eq!(
            StackChecker::determine_status(&[], false),
            CrateStatus::Healthy
        );

        // Info only = healthy
        let info_issue = CrateIssue::new(IssueSeverity::Info, IssueType::NotPublished, "test");
        assert_eq!(
            StackChecker::determine_status(&[info_issue], false),
            CrateStatus::Healthy
        );

        // Warning = warning (non-strict)
        let warning_issue =
            CrateIssue::new(IssueSeverity::Warning, IssueType::VersionBehind, "test");
        assert_eq!(
            StackChecker::determine_status(std::slice::from_ref(&warning_issue), false),
            CrateStatus::Warning
        );

        // Warning = error (strict)
        assert_eq!(
            StackChecker::determine_status(&[warning_issue], true),
            CrateStatus::Error
        );

        // Error = error
        let error_issue = CrateIssue::new(IssueSeverity::Error, IssueType::PathDependency, "test");
        assert_eq!(
            StackChecker::determine_status(&[error_issue], false),
            CrateStatus::Error
        );
    }
}
