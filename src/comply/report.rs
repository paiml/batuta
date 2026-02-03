//! Stack Compliance Report Generation
//!
//! Generates reports in multiple formats: Text, JSON, HTML, Markdown.

use crate::comply::rule::{FixResult, RuleResult, RuleViolation, ViolationLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Report output format
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ComplyReportFormat {
    #[default]
    Text,
    Json,
    Markdown,
    Html,
}

/// Stack compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplyReport {
    /// Results per project per rule
    pub results: HashMap<String, HashMap<String, ProjectRuleResult>>,
    /// Exemptions applied
    pub exemptions: Vec<Exemption>,
    /// Global errors
    pub errors: Vec<String>,
    /// Summary statistics
    pub summary: ComplianceSummary,
    /// Whether report has been finalized
    #[serde(skip)]
    finalized: bool,
}

/// Result for a specific project-rule pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectRuleResult {
    /// Rule check result
    Checked(RuleResult),
    /// Rule was exempt
    Exempt(String),
    /// Error occurred during check
    Error(String),
    /// Fix was applied
    Fixed(FixResult),
    /// Dry-run fix preview
    DryRunFix(Vec<RuleViolation>),
}

/// Record of an exemption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exemption {
    pub project: String,
    pub rule: String,
    pub reason: Option<String>,
}

/// Summary statistics for the report
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// Total projects checked
    pub total_projects: usize,
    /// Projects that passed all rules
    pub passing_projects: usize,
    /// Projects with violations
    pub failing_projects: usize,
    /// Total rules checked
    pub total_checks: usize,
    /// Checks that passed
    pub passed_checks: usize,
    /// Checks that failed
    pub failed_checks: usize,
    /// Total violations found
    pub total_violations: usize,
    /// Violations by severity
    pub violations_by_severity: HashMap<String, usize>,
    /// Fixable violations
    pub fixable_violations: usize,
    /// Pass rate as percentage
    pub pass_rate: f64,
}

/// A single violation for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub project: String,
    pub rule: String,
    pub code: String,
    pub message: String,
    pub severity: ViolationSeverity,
    pub location: Option<String>,
    pub fixable: bool,
}

/// Violation severity for display
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl From<ViolationLevel> for ViolationSeverity {
    fn from(level: ViolationLevel) -> Self {
        match level {
            ViolationLevel::Info => ViolationSeverity::Info,
            ViolationLevel::Warning => ViolationSeverity::Warning,
            ViolationLevel::Error => ViolationSeverity::Error,
            ViolationLevel::Critical => ViolationSeverity::Critical,
        }
    }
}

impl ComplyReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            exemptions: Vec::new(),
            errors: Vec::new(),
            summary: ComplianceSummary::default(),
            finalized: false,
        }
    }

    /// Add a rule result for a project
    pub fn add_result(&mut self, project: &str, rule: &str, result: RuleResult) {
        self.results
            .entry(project.to_string())
            .or_default()
            .insert(rule.to_string(), ProjectRuleResult::Checked(result));
    }

    /// Add an exemption
    pub fn add_exemption(&mut self, project: &str, rule: &str) {
        self.results
            .entry(project.to_string())
            .or_default()
            .insert(rule.to_string(), ProjectRuleResult::Exempt(rule.to_string()));
        self.exemptions.push(Exemption {
            project: project.to_string(),
            rule: rule.to_string(),
            reason: None,
        });
    }

    /// Add an error
    pub fn add_error(&mut self, project: &str, rule: &str, error: String) {
        self.results
            .entry(project.to_string())
            .or_default()
            .insert(rule.to_string(), ProjectRuleResult::Error(error));
    }

    /// Add a global error
    pub fn add_global_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a fix result
    pub fn add_fix_result(&mut self, project: &str, rule: &str, result: FixResult) {
        self.results
            .entry(project.to_string())
            .or_default()
            .insert(rule.to_string(), ProjectRuleResult::Fixed(result));
    }

    /// Add a dry-run fix preview
    pub fn add_dry_run_fix(&mut self, project: &str, rule: &str, violations: &[RuleViolation]) {
        self.results
            .entry(project.to_string())
            .or_default()
            .insert(
                rule.to_string(),
                ProjectRuleResult::DryRunFix(violations.to_vec()),
            );
    }

    /// Finalize the report and compute summary
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }

        let mut total_projects = 0;
        let mut passing_projects = 0;
        let mut total_checks = 0;
        let mut passed_checks = 0;
        let mut failed_checks = 0;
        let mut total_violations = 0;
        let mut fixable_violations = 0;
        let mut violations_by_severity: HashMap<String, usize> = HashMap::new();

        for rules in self.results.values() {
            total_projects += 1;
            let mut project_passed = true;

            for result in rules.values() {
                total_checks += 1;

                match result {
                    ProjectRuleResult::Checked(r) => {
                        if r.passed {
                            passed_checks += 1;
                        } else {
                            failed_checks += 1;
                            project_passed = false;

                            for v in &r.violations {
                                total_violations += 1;
                                if v.fixable {
                                    fixable_violations += 1;
                                }
                                *violations_by_severity
                                    .entry(format!("{}", v.severity))
                                    .or_default() += 1;
                            }
                        }
                    }
                    ProjectRuleResult::Exempt(_) => {
                        passed_checks += 1;
                    }
                    ProjectRuleResult::Error(_) => {
                        failed_checks += 1;
                        project_passed = false;
                    }
                    ProjectRuleResult::Fixed(r) => {
                        if r.success {
                            passed_checks += 1;
                        } else {
                            failed_checks += 1;
                            project_passed = false;
                        }
                    }
                    ProjectRuleResult::DryRunFix(violations) => {
                        failed_checks += 1;
                        project_passed = false;
                        total_violations += violations.len();
                        for v in violations {
                            if v.fixable {
                                fixable_violations += 1;
                            }
                        }
                    }
                }
            }

            if project_passed {
                passing_projects += 1;
            }
        }

        let pass_rate = if total_checks > 0 {
            (passed_checks as f64 / total_checks as f64) * 100.0
        } else {
            100.0
        };

        self.summary = ComplianceSummary {
            total_projects,
            passing_projects,
            failing_projects: total_projects - passing_projects,
            total_checks,
            passed_checks,
            failed_checks,
            total_violations,
            violations_by_severity,
            fixable_violations,
            pass_rate,
        };

        self.finalized = true;
    }

    /// Get all violations as a flat list
    pub fn violations(&self) -> Vec<Violation> {
        let mut violations = Vec::new();

        for (project, rules) in &self.results {
            for (rule, result) in rules {
                if let ProjectRuleResult::Checked(r) = result {
                    for v in &r.violations {
                        violations.push(Violation {
                            project: project.clone(),
                            rule: rule.clone(),
                            code: v.code.clone(),
                            message: v.message.clone(),
                            severity: v.severity.into(),
                            location: v.location.clone(),
                            fixable: v.fixable,
                        });
                    }
                }
            }
        }

        violations
    }

    /// Check if the report indicates overall compliance
    pub fn is_compliant(&self) -> bool {
        self.summary.failing_projects == 0 && self.errors.is_empty()
    }

    /// Format as text
    pub fn format_text(&self) -> String {
        let mut out = String::new();

        writeln!(out, "STACK COMPLIANCE REPORT").unwrap();
        writeln!(out, "=======================\n").unwrap();

        // Summary
        writeln!(
            out,
            "Projects: {}/{} passing ({:.1}%)",
            self.summary.passing_projects, self.summary.total_projects, self.summary.pass_rate
        )
        .unwrap();
        writeln!(out, "Violations: {}", self.summary.total_violations).unwrap();
        if self.summary.fixable_violations > 0 {
            writeln!(
                out,
                "Fixable: {} ({:.1}%)",
                self.summary.fixable_violations,
                (self.summary.fixable_violations as f64 / self.summary.total_violations as f64)
                    * 100.0
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        // Per-project results
        for (project, rules) in &self.results {
            let passed = rules
                .values()
                .all(|r| matches!(r, ProjectRuleResult::Checked(r) if r.passed) || matches!(r, ProjectRuleResult::Exempt(_)));

            let status = if passed { "PASS" } else { "FAIL" };
            writeln!(out, "{} {} {}", project, ".".repeat(40 - project.len().min(39)), status).unwrap();

            for (rule, result) in rules {
                match result {
                    ProjectRuleResult::Checked(r) => {
                        if !r.passed {
                            for v in &r.violations {
                                writeln!(
                                    out,
                                    "  [{:?}] {}: {}",
                                    v.severity, v.code, v.message
                                )
                                .unwrap();
                                if let Some(loc) = &v.location {
                                    writeln!(out, "         at {}", loc).unwrap();
                                }
                            }
                        }
                    }
                    ProjectRuleResult::Exempt(reason) => {
                        writeln!(out, "  [EXEMPT] {} - {}", rule, reason).unwrap();
                    }
                    ProjectRuleResult::Error(e) => {
                        writeln!(out, "  [ERROR] {} - {}", rule, e).unwrap();
                    }
                    ProjectRuleResult::Fixed(r) => {
                        writeln!(out, "  [FIXED] {} fixes applied", r.fixed_count).unwrap();
                    }
                    ProjectRuleResult::DryRunFix(violations) => {
                        writeln!(
                            out,
                            "  [DRY-RUN] {} violations would be fixed",
                            violations.len()
                        )
                        .unwrap();
                    }
                }
            }
        }

        // Global errors
        if !self.errors.is_empty() {
            writeln!(out, "\nGlobal Errors:").unwrap();
            for e in &self.errors {
                writeln!(out, "  - {}", e).unwrap();
            }
        }

        out
    }

    /// Format as JSON
    pub fn format_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format as Markdown
    pub fn format_markdown(&self) -> String {
        let mut out = String::new();

        writeln!(out, "# Stack Compliance Report\n").unwrap();

        writeln!(out, "## Summary\n").unwrap();
        writeln!(out, "| Metric | Value |").unwrap();
        writeln!(out, "|--------|-------|").unwrap();
        writeln!(
            out,
            "| Projects Passing | {}/{} ({:.1}%) |",
            self.summary.passing_projects, self.summary.total_projects, self.summary.pass_rate
        )
        .unwrap();
        writeln!(out, "| Total Violations | {} |", self.summary.total_violations).unwrap();
        writeln!(
            out,
            "| Fixable Violations | {} |",
            self.summary.fixable_violations
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "## Results by Project\n").unwrap();

        for (project, rules) in &self.results {
            let passed = rules
                .values()
                .all(|r| matches!(r, ProjectRuleResult::Checked(r) if r.passed));
            let emoji = if passed { "✅" } else { "❌" };

            writeln!(out, "### {} {}\n", emoji, project).unwrap();

            for (rule, result) in rules {
                match result {
                    ProjectRuleResult::Checked(r) => {
                        if r.passed {
                            writeln!(out, "- ✅ **{}**: Passed", rule).unwrap();
                        } else {
                            writeln!(out, "- ❌ **{}**: {} violations", rule, r.violations.len())
                                .unwrap();
                            for v in &r.violations {
                                writeln!(out, "  - `{}`: {}", v.code, v.message).unwrap();
                            }
                        }
                    }
                    ProjectRuleResult::Exempt(reason) => {
                        writeln!(out, "- ⏭️ **{}**: Exempt - {}", rule, reason).unwrap();
                    }
                    ProjectRuleResult::Error(e) => {
                        writeln!(out, "- ⚠️ **{}**: Error - {}", rule, e).unwrap();
                    }
                    _ => {}
                }
            }
            writeln!(out).unwrap();
        }

        out
    }

    /// Format report based on format type
    pub fn format(&self, format: ComplyReportFormat) -> String {
        match format {
            ComplyReportFormat::Text => self.format_text(),
            ComplyReportFormat::Json => self.format_json(),
            ComplyReportFormat::Markdown => self.format_markdown(),
            ComplyReportFormat::Html => self.format_html(),
        }
    }

    /// Format as HTML
    pub fn format_html(&self) -> String {
        let mut out = String::new();

        writeln!(out, r#"<!DOCTYPE html>
<html>
<head>
    <title>Stack Compliance Report</title>
    <style>
        body {{ font-family: Roboto, sans-serif; margin: 40px; }}
        .pass {{ color: #34A853; }}
        .fail {{ color: #EA4335; }}
        .warn {{ color: #FBBC04; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #6750A4; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Stack Compliance Report</h1>

    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Projects</td><td>{}/{} ({:.1}%)</td></tr>
        <tr><td>Total Violations</td><td>{}</td></tr>
        <tr><td>Fixable</td><td>{}</td></tr>
    </table>
"#,
            self.summary.passing_projects,
            self.summary.total_projects,
            self.summary.pass_rate,
            self.summary.total_violations,
            self.summary.fixable_violations
        ).unwrap();

        writeln!(out, "    <h2>Results</h2>").unwrap();
        writeln!(out, "    <table>").unwrap();
        writeln!(out, "        <tr><th>Project</th><th>Status</th><th>Violations</th></tr>").unwrap();

        for (project, rules) in &self.results {
            let passed = rules
                .values()
                .all(|r| matches!(r, ProjectRuleResult::Checked(r) if r.passed));
            let status_class = if passed { "pass" } else { "fail" };
            let status = if passed { "PASS" } else { "FAIL" };

            let violation_count: usize = rules
                .values()
                .filter_map(|r| match r {
                    ProjectRuleResult::Checked(r) => Some(r.violations.len()),
                    _ => None,
                })
                .sum();

            writeln!(
                out,
                "        <tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td></tr>",
                project, status_class, status, violation_count
            )
            .unwrap();
        }

        writeln!(out, "    </table>").unwrap();
        writeln!(out, "</body></html>").unwrap();

        out
    }
}

impl Default for ComplyReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comply::rule::RuleResult;

    #[test]
    fn test_report_creation() {
        let report = ComplyReport::new();
        assert!(report.results.is_empty());
        assert!(!report.finalized);
    }

    #[test]
    fn test_add_result() {
        let mut report = ComplyReport::new();
        report.add_result("test-project", "test-rule", RuleResult::pass());
        assert!(report.results.contains_key("test-project"));
    }

    #[test]
    fn test_finalize() {
        let mut report = ComplyReport::new();
        report.add_result("project1", "rule1", RuleResult::pass());
        report.add_result("project2", "rule1", RuleResult::fail(vec![]));
        report.finalize();

        assert!(report.finalized);
        assert_eq!(report.summary.total_projects, 2);
        assert_eq!(report.summary.passing_projects, 1);
    }

    #[test]
    fn test_format_text() {
        let mut report = ComplyReport::new();
        report.add_result("trueno", "makefile-targets", RuleResult::pass());
        report.finalize();

        let text = report.format_text();
        assert!(text.contains("STACK COMPLIANCE REPORT"));
        assert!(text.contains("trueno"));
    }

    #[test]
    fn test_format_json() {
        let mut report = ComplyReport::new();
        report.add_result("trueno", "makefile-targets", RuleResult::pass());
        report.finalize();

        let json = report.format_json();
        assert!(json.contains("trueno"));
        assert!(json.contains("summary"));
    }

    #[test]
    fn test_is_compliant() {
        let mut report = ComplyReport::new();
        report.add_result("project1", "rule1", RuleResult::pass());
        report.finalize();
        assert!(report.is_compliant());

        let mut report2 = ComplyReport::new();
        report2.add_result("project1", "rule1", RuleResult::fail(vec![]));
        report2.finalize();
        assert!(!report2.is_compliant());
    }
}
