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
    ///
    /// Note: writeln! to String is infallible (fmt::Write impl for String
    /// always returns Ok).
    pub fn format_text(&self) -> String {
        let mut out = String::new();

        writeln!(out, "STACK COMPLIANCE REPORT").ok();
        writeln!(out, "=======================\n").ok();

        // Summary
        writeln!(
            out,
            "Projects: {}/{} passing ({:.1}%)",
            self.summary.passing_projects, self.summary.total_projects, self.summary.pass_rate
        )
        .ok();
        writeln!(out, "Violations: {}", self.summary.total_violations).ok();
        if self.summary.fixable_violations > 0 {
            writeln!(
                out,
                "Fixable: {} ({:.1}%)",
                self.summary.fixable_violations,
                (self.summary.fixable_violations as f64 / self.summary.total_violations as f64)
                    * 100.0
            )
            .ok();
        }
        writeln!(out).ok();

        // Per-project results
        for (project, rules) in &self.results {
            let passed = rules
                .values()
                .all(|r| matches!(r, ProjectRuleResult::Checked(r) if r.passed) || matches!(r, ProjectRuleResult::Exempt(_)));

            let status = if passed { "PASS" } else { "FAIL" };
            writeln!(out, "{} {} {}", project, ".".repeat(40 - project.len().min(39)), status).ok();

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
                                .ok();
                                if let Some(loc) = &v.location {
                                    writeln!(out, "         at {}", loc).ok();
                                }
                            }
                        }
                    }
                    ProjectRuleResult::Exempt(reason) => {
                        writeln!(out, "  [EXEMPT] {} - {}", rule, reason).ok();
                    }
                    ProjectRuleResult::Error(e) => {
                        writeln!(out, "  [ERROR] {} - {}", rule, e).ok();
                    }
                    ProjectRuleResult::Fixed(r) => {
                        writeln!(out, "  [FIXED] {} fixes applied", r.fixed_count).ok();
                    }
                    ProjectRuleResult::DryRunFix(violations) => {
                        writeln!(
                            out,
                            "  [DRY-RUN] {} violations would be fixed",
                            violations.len()
                        )
                        .ok();
                    }
                }
            }
        }

        // Global errors
        if !self.errors.is_empty() {
            writeln!(out, "\nGlobal Errors:").ok();
            for e in &self.errors {
                writeln!(out, "  - {}", e).ok();
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

        writeln!(out, "# Stack Compliance Report\n").ok();

        writeln!(out, "## Summary\n").ok();
        writeln!(out, "| Metric | Value |").ok();
        writeln!(out, "|--------|-------|").ok();
        writeln!(
            out,
            "| Projects Passing | {}/{} ({:.1}%) |",
            self.summary.passing_projects, self.summary.total_projects, self.summary.pass_rate
        )
        .ok();
        writeln!(out, "| Total Violations | {} |", self.summary.total_violations).ok();
        writeln!(
            out,
            "| Fixable Violations | {} |",
            self.summary.fixable_violations
        )
        .ok();
        writeln!(out).ok();

        writeln!(out, "## Results by Project\n").ok();

        for (project, rules) in &self.results {
            let passed = rules
                .values()
                .all(|r| matches!(r, ProjectRuleResult::Checked(r) if r.passed));
            let emoji = if passed { "✅" } else { "❌" };

            writeln!(out, "### {} {}\n", emoji, project).ok();

            for (rule, result) in rules {
                match result {
                    ProjectRuleResult::Checked(r) => {
                        if r.passed {
                            writeln!(out, "- ✅ **{}**: Passed", rule).ok();
                        } else {
                            writeln!(out, "- ❌ **{}**: {} violations", rule, r.violations.len())
                                .ok();
                            for v in &r.violations {
                                writeln!(out, "  - `{}`: {}", v.code, v.message).ok();
                            }
                        }
                    }
                    ProjectRuleResult::Exempt(reason) => {
                        writeln!(out, "- ⏭️ **{}**: Exempt - {}", rule, reason).ok();
                    }
                    ProjectRuleResult::Error(e) => {
                        writeln!(out, "- ⚠️ **{}**: Error - {}", rule, e).ok();
                    }
                    _ => {}
                }
            }
            writeln!(out).ok();
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
        ).ok();

        writeln!(out, "    <h2>Results</h2>").ok();
        writeln!(out, "    <table>").ok();
        writeln!(out, "        <tr><th>Project</th><th>Status</th><th>Violations</th></tr>").ok();

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
            .ok();
        }

        writeln!(out, "    </table>").ok();
        writeln!(out, "</body></html>").ok();

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

    #[test]
    fn test_add_exemption() {
        let mut report = ComplyReport::new();
        report.add_exemption("project", "rule");
        assert_eq!(report.exemptions.len(), 1);
        assert_eq!(report.exemptions[0].project, "project");
        assert_eq!(report.exemptions[0].rule, "rule");
    }

    #[test]
    fn test_add_error() {
        let mut report = ComplyReport::new();
        report.add_error("project", "rule", "Error message".to_string());
        let result = report.results.get("project").unwrap().get("rule").unwrap();
        assert!(matches!(result, ProjectRuleResult::Error(_)));
    }

    #[test]
    fn test_add_global_error() {
        let mut report = ComplyReport::new();
        report.add_global_error("Global error".to_string());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0], "Global error");
    }

    #[test]
    fn test_add_fix_result() {
        let mut report = ComplyReport::new();
        report.add_fix_result("project", "rule", FixResult::success(5));
        let result = report.results.get("project").unwrap().get("rule").unwrap();
        assert!(matches!(result, ProjectRuleResult::Fixed(_)));
    }

    #[test]
    fn test_add_dry_run_fix() {
        let mut report = ComplyReport::new();
        let violations = vec![RuleViolation::new("V-001", "Test violation")];
        report.add_dry_run_fix("project", "rule", &violations);
        let result = report.results.get("project").unwrap().get("rule").unwrap();
        assert!(matches!(result, ProjectRuleResult::DryRunFix(_)));
    }

    #[test]
    fn test_violations() {
        let mut report = ComplyReport::new();
        let violations = vec![RuleViolation::new("V-001", "Test violation")];
        report.add_result("project", "rule", RuleResult::fail(violations));
        report.finalize();
        let vs = report.violations();
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0].code, "V-001");
    }

    #[test]
    fn test_format_markdown() {
        let mut report = ComplyReport::new();
        report.add_result("project", "rule", RuleResult::pass());
        report.finalize();
        let md = report.format_markdown();
        assert!(md.contains("# Stack Compliance Report"));
        assert!(md.contains("project"));
    }

    #[test]
    fn test_format_html() {
        let mut report = ComplyReport::new();
        report.add_result("project", "rule", RuleResult::pass());
        report.finalize();
        let html = report.format_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Stack Compliance Report"));
    }

    #[test]
    fn test_format_dispatch() {
        let mut report = ComplyReport::new();
        report.add_result("project", "rule", RuleResult::pass());
        report.finalize();

        let text = report.format(ComplyReportFormat::Text);
        assert!(text.contains("STACK COMPLIANCE REPORT"));

        let json = report.format(ComplyReportFormat::Json);
        assert!(json.contains("summary"));

        let md = report.format(ComplyReportFormat::Markdown);
        assert!(md.contains("# Stack"));

        let html = report.format(ComplyReportFormat::Html);
        assert!(html.contains("<html>"));
    }

    #[test]
    fn test_violation_severity_from() {
        assert!(matches!(ViolationSeverity::from(ViolationLevel::Info), ViolationSeverity::Info));
        assert!(matches!(ViolationSeverity::from(ViolationLevel::Warning), ViolationSeverity::Warning));
        assert!(matches!(ViolationSeverity::from(ViolationLevel::Error), ViolationSeverity::Error));
        assert!(matches!(ViolationSeverity::from(ViolationLevel::Critical), ViolationSeverity::Critical));
    }

    #[test]
    fn test_compliance_summary_default() {
        let summary = ComplianceSummary::default();
        assert_eq!(summary.total_projects, 0);
        assert_eq!(summary.total_violations, 0);
        assert_eq!(summary.pass_rate, 0.0);
    }

    #[test]
    fn test_report_default() {
        let report = ComplyReport::default();
        assert!(report.results.is_empty());
    }

    #[test]
    fn test_finalize_idempotent() {
        let mut report = ComplyReport::new();
        report.add_result("project", "rule", RuleResult::pass());
        report.finalize();
        let summary1 = report.summary.clone();
        report.finalize(); // Should not change
        assert_eq!(report.summary.total_projects, summary1.total_projects);
    }

    #[test]
    fn test_exemption_fields() {
        let exemption = Exemption {
            project: "test-project".to_string(),
            rule: "test-rule".to_string(),
            reason: Some("Legacy code".to_string()),
        };
        assert_eq!(exemption.project, "test-project");
        assert_eq!(exemption.rule, "test-rule");
        assert_eq!(exemption.reason, Some("Legacy code".to_string()));
    }

    #[test]
    fn test_exemption_clone() {
        let exemption = Exemption {
            project: "p".to_string(),
            rule: "r".to_string(),
            reason: None,
        };
        let cloned = exemption.clone();
        assert_eq!(cloned.project, exemption.project);
    }

    #[test]
    fn test_violation_fields() {
        let v = Violation {
            project: "proj".to_string(),
            rule: "rule".to_string(),
            code: "V-001".to_string(),
            message: "msg".to_string(),
            severity: ViolationSeverity::Error,
            location: Some("file.rs:10".to_string()),
            fixable: true,
        };
        assert_eq!(v.project, "proj");
        assert_eq!(v.code, "V-001");
        assert!(v.location.is_some());
        assert!(v.fixable);
    }

    #[test]
    fn test_violation_clone() {
        let v = Violation {
            project: "p".to_string(),
            rule: "r".to_string(),
            code: "C".to_string(),
            message: "m".to_string(),
            severity: ViolationSeverity::Warning,
            location: None,
            fixable: false,
        };
        let cloned = v.clone();
        assert_eq!(cloned.code, v.code);
        assert!(!cloned.fixable);
    }

    #[test]
    fn test_violation_severity_debug() {
        let info = ViolationSeverity::Info;
        let warning = ViolationSeverity::Warning;
        let error = ViolationSeverity::Error;
        let critical = ViolationSeverity::Critical;

        assert!(format!("{:?}", info).contains("Info"));
        assert!(format!("{:?}", warning).contains("Warning"));
        assert!(format!("{:?}", error).contains("Error"));
        assert!(format!("{:?}", critical).contains("Critical"));
    }

    #[test]
    fn test_comply_report_format_default() {
        let fmt = ComplyReportFormat::default();
        assert!(matches!(fmt, ComplyReportFormat::Text));
    }

    #[test]
    fn test_comply_report_format_clone() {
        let fmt = ComplyReportFormat::Json;
        let cloned = fmt;
        assert!(matches!(cloned, ComplyReportFormat::Json));
    }

    #[test]
    fn test_finalize_with_exemptions() {
        let mut report = ComplyReport::new();
        report.add_exemption("project", "rule");
        report.finalize();
        assert_eq!(report.summary.passed_checks, 1);
    }

    #[test]
    fn test_finalize_with_errors() {
        let mut report = ComplyReport::new();
        report.add_error("project", "rule", "Error".to_string());
        report.finalize();
        assert_eq!(report.summary.failed_checks, 1);
        assert_eq!(report.summary.failing_projects, 1);
    }

    #[test]
    fn test_finalize_with_fix_success() {
        let mut report = ComplyReport::new();
        report.add_fix_result("project", "rule", FixResult::success(3));
        report.finalize();
        assert_eq!(report.summary.passed_checks, 1);
    }

    #[test]
    fn test_finalize_with_fix_failure() {
        let mut report = ComplyReport::new();
        report.add_fix_result("project", "rule", FixResult::failure("Fix failed"));
        report.finalize();
        assert_eq!(report.summary.failed_checks, 1);
    }

    #[test]
    fn test_finalize_with_dry_run_fix() {
        let mut report = ComplyReport::new();
        let violations = vec![
            RuleViolation::new("V-001", "Test").fixable(),
            RuleViolation::new("V-002", "Test2"), // not fixable
        ];
        report.add_dry_run_fix("project", "rule", &violations);
        report.finalize();
        assert_eq!(report.summary.total_violations, 2);
        assert_eq!(report.summary.fixable_violations, 1);
    }

    #[test]
    fn test_finalize_with_fixable_violations() {
        let mut report = ComplyReport::new();
        let violations = vec![
            RuleViolation::new("V-001", "Test").fixable(),
        ];
        report.add_result("project", "rule", RuleResult::fail(violations));
        report.finalize();
        assert_eq!(report.summary.fixable_violations, 1);
    }

    #[test]
    fn test_finalize_violations_by_severity() {
        let mut report = ComplyReport::new();
        let mut v = RuleViolation::new("V-001", "Error violation");
        v.severity = ViolationLevel::Error;
        report.add_result("project", "rule", RuleResult::fail(vec![v]));
        report.finalize();
        assert!(report.summary.violations_by_severity.contains_key("ERROR"));
    }

    #[test]
    fn test_format_text_with_violations() {
        let mut report = ComplyReport::new();
        let v = RuleViolation::new("V-001", "Test violation")
            .with_location("file.rs:10");
        report.add_result("project", "rule", RuleResult::fail(vec![v]));
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("V-001"));
        assert!(text.contains("Test violation"));
        assert!(text.contains("at file.rs:10"));
    }

    #[test]
    fn test_format_text_with_exempt() {
        let mut report = ComplyReport::new();
        report.add_exemption("project", "rule");
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("[EXEMPT]"));
    }

    #[test]
    fn test_format_text_with_error() {
        let mut report = ComplyReport::new();
        report.add_error("project", "rule", "Something went wrong".to_string());
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("[ERROR]"));
        assert!(text.contains("Something went wrong"));
    }

    #[test]
    fn test_format_text_with_fixed() {
        let mut report = ComplyReport::new();
        report.add_fix_result("project", "rule", FixResult::success(5));
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("[FIXED]"));
        assert!(text.contains("5 fixes applied"));
    }

    #[test]
    fn test_format_text_with_dry_run() {
        let mut report = ComplyReport::new();
        report.add_dry_run_fix("project", "rule", &[RuleViolation::new("V-001", "Test")]);
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("[DRY-RUN]"));
    }

    #[test]
    fn test_format_text_with_global_errors() {
        let mut report = ComplyReport::new();
        report.add_global_error("Global error 1".to_string());
        report.add_global_error("Global error 2".to_string());
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("Global Errors:"));
        assert!(text.contains("Global error 1"));
    }

    #[test]
    fn test_format_text_fixable_stats() {
        let mut report = ComplyReport::new();
        let v = RuleViolation::new("V-001", "Test").fixable();
        report.add_result("project", "rule", RuleResult::fail(vec![v]));
        report.finalize();
        let text = report.format_text();
        assert!(text.contains("Fixable:"));
    }

    #[test]
    fn test_format_markdown_with_exempt() {
        let mut report = ComplyReport::new();
        report.add_exemption("project", "rule");
        report.finalize();
        let md = report.format_markdown();
        assert!(md.contains("Exempt"));
    }

    #[test]
    fn test_format_markdown_with_error() {
        let mut report = ComplyReport::new();
        report.add_error("project", "rule", "Error msg".to_string());
        report.finalize();
        let md = report.format_markdown();
        assert!(md.contains("Error"));
    }

    #[test]
    fn test_format_markdown_with_violations() {
        let mut report = ComplyReport::new();
        let v = RuleViolation::new("V-001", "Test violation");
        report.add_result("project", "rule", RuleResult::fail(vec![v]));
        report.finalize();
        let md = report.format_markdown();
        assert!(md.contains("V-001"));
    }

    #[test]
    fn test_format_html_with_violations() {
        let mut report = ComplyReport::new();
        let v = RuleViolation::new("V-001", "Test");
        report.add_result("project", "rule", RuleResult::fail(vec![v]));
        report.finalize();
        let html = report.format_html();
        assert!(html.contains("project"));
        assert!(html.contains("FAIL"));
    }

    #[test]
    fn test_is_compliant_with_errors() {
        let mut report = ComplyReport::new();
        report.add_result("project", "rule", RuleResult::pass());
        report.add_global_error("Error".to_string());
        report.finalize();
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_compliance_summary_fields() {
        let summary = ComplianceSummary {
            total_projects: 10,
            passing_projects: 8,
            failing_projects: 2,
            total_checks: 40,
            passed_checks: 35,
            failed_checks: 5,
            total_violations: 15,
            violations_by_severity: HashMap::new(),
            fixable_violations: 10,
            pass_rate: 87.5,
        };
        assert_eq!(summary.total_projects, 10);
        assert_eq!(summary.pass_rate, 87.5);
    }

    #[test]
    fn test_compliance_summary_clone() {
        let summary = ComplianceSummary::default();
        let cloned = summary.clone();
        assert_eq!(cloned.total_projects, summary.total_projects);
    }

    #[test]
    fn test_project_rule_result_variants() {
        let checked = ProjectRuleResult::Checked(RuleResult::pass());
        let exempt = ProjectRuleResult::Exempt("reason".to_string());
        let error = ProjectRuleResult::Error("error".to_string());
        let fixed = ProjectRuleResult::Fixed(FixResult::success(1));
        let dry_run = ProjectRuleResult::DryRunFix(vec![]);

        assert!(matches!(checked, ProjectRuleResult::Checked(_)));
        assert!(matches!(exempt, ProjectRuleResult::Exempt(_)));
        assert!(matches!(error, ProjectRuleResult::Error(_)));
        assert!(matches!(fixed, ProjectRuleResult::Fixed(_)));
        assert!(matches!(dry_run, ProjectRuleResult::DryRunFix(_)));
    }

    #[test]
    fn test_violations_empty() {
        let report = ComplyReport::new();
        assert!(report.violations().is_empty());
    }

    #[test]
    fn test_finalize_zero_checks_pass_rate() {
        let mut report = ComplyReport::new();
        report.finalize();
        assert_eq!(report.summary.pass_rate, 100.0);
    }
}
