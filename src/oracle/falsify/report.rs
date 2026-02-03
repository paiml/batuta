//! Falsification Report Generation
//!
//! Generates reports from test execution results.

use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;

/// Test outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    /// Test passed (claim not falsified)
    Passed,
    /// Test failed (claim falsified - this is good!)
    Falsified,
    /// Test skipped
    Skipped,
    /// Test errored (infrastructure issue)
    Error,
}

/// Falsification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationReport {
    /// Spec name
    pub spec_name: String,
    /// Test results
    pub results: Vec<TestResult>,
    /// Summary statistics
    pub summary: FalsificationSummary,
    /// Generation timestamp
    pub generated_at: String,
}

/// Single test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test ID
    pub id: String,
    /// Test name
    pub name: String,
    /// Category
    pub category: String,
    /// Points
    pub points: u32,
    /// Outcome
    pub outcome: TestOutcome,
    /// Error message if any
    pub error: Option<String>,
    /// Evidence collected
    pub evidence: Vec<String>,
}

/// Summary statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FalsificationSummary {
    /// Total tests
    pub total_tests: usize,
    /// Total points
    pub total_points: u32,
    /// Tests passed
    pub passed: usize,
    /// Tests falsified (failures = success in Popperian methodology!)
    pub falsified: usize,
    /// Tests skipped
    pub skipped: usize,
    /// Tests errored
    pub errors: usize,
    /// Falsification rate (target: 5-15%)
    pub falsification_rate: f64,
    /// Points by category
    pub points_by_category: std::collections::HashMap<String, CategoryStats>,
}

/// Category statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryStats {
    pub total: u32,
    pub passed: u32,
    pub falsified: u32,
}

impl FalsificationReport {
    /// Create a new report
    pub fn new(spec_name: String) -> Self {
        Self {
            spec_name,
            results: Vec::new(),
            summary: FalsificationSummary::default(),
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add a test result
    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    /// Finalize and compute summary
    pub fn finalize(&mut self) {
        self.summary.total_tests = self.results.len();
        self.summary.total_points = self.results.iter().map(|r| r.points).sum();

        for result in &self.results {
            match result.outcome {
                TestOutcome::Passed => self.summary.passed += 1,
                TestOutcome::Falsified => self.summary.falsified += 1,
                TestOutcome::Skipped => self.summary.skipped += 1,
                TestOutcome::Error => self.summary.errors += 1,
            }

            let entry = self
                .summary
                .points_by_category
                .entry(result.category.clone())
                .or_default();
            entry.total += result.points;
            match result.outcome {
                TestOutcome::Passed => entry.passed += result.points,
                TestOutcome::Falsified => entry.falsified += result.points,
                _ => {}
            }
        }

        let executed = self.summary.passed + self.summary.falsified;
        if executed > 0 {
            self.summary.falsification_rate = (self.summary.falsified as f64) / (executed as f64);
        }
    }

    /// Format as markdown
    pub fn format_markdown(&self) -> String {
        let mut out = String::new();

        writeln!(out, "# Falsification Report: {}", self.spec_name).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "**Generated**: {}", self.generated_at).unwrap();
        writeln!(out, "**Total Points**: {}", self.summary.total_points).unwrap();
        writeln!(
            out,
            "**Falsifications Found**: {} (target: 5-15%)",
            self.summary.falsified
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "## Summary").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| Category | Points | Passed | Failed | Pass Rate |").unwrap();
        writeln!(out, "|----------|--------|--------|--------|-----------|").unwrap();

        for (category, stats) in &self.summary.points_by_category {
            let pass_rate = if stats.total > 0 {
                (stats.passed as f64 / stats.total as f64) * 100.0
            } else {
                0.0
            };
            writeln!(
                out,
                "| {} | {} | {} | {} | {:.0}% |",
                category, stats.total, stats.passed, stats.falsified, pass_rate
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        // Verdict
        let verdict = if self.summary.falsification_rate >= 0.05
            && self.summary.falsification_rate <= 0.15
        {
            "Healthy falsification rate - specification is well-tested"
        } else if self.summary.falsification_rate < 0.05 {
            "Low falsification rate - consider more edge cases"
        } else {
            "High falsification rate - specification needs hardening"
        };

        writeln!(
            out,
            "**Verdict**: {:.1}% falsification rate - {}",
            self.summary.falsification_rate * 100.0,
            verdict
        )
        .unwrap();
        writeln!(out).unwrap();

        // Falsifications (failures = success!)
        if self.summary.falsified > 0 {
            writeln!(out, "## Falsifications (Failures = Success!)").unwrap();
            writeln!(out).unwrap();

            for result in &self.results {
                if result.outcome == TestOutcome::Falsified {
                    writeln!(out, "### {}: {}", result.id, result.name).unwrap();
                    writeln!(out, "**Status**: FALSIFIED").unwrap();
                    writeln!(out, "**Points**: {}", result.points).unwrap();
                    if let Some(err) = &result.error {
                        writeln!(out, "**Details**: {}", err).unwrap();
                    }
                    for evidence in &result.evidence {
                        writeln!(out, "- {}", evidence).unwrap();
                    }
                    writeln!(out).unwrap();
                }
            }
        }

        out
    }

    /// Format as JSON
    pub fn format_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format as text
    pub fn format_text(&self) -> String {
        let mut out = String::new();

        writeln!(out, "FALSIFICATION REPORT: {}", self.spec_name).unwrap();
        writeln!(out, "{}", "=".repeat(60)).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "Total Points: {}", self.summary.total_points).unwrap();
        writeln!(
            out,
            "Falsifications: {} ({:.1}%)",
            self.summary.falsified,
            self.summary.falsification_rate * 100.0
        )
        .unwrap();
        writeln!(out).unwrap();

        for result in &self.results {
            let status = match result.outcome {
                TestOutcome::Passed => "PASS",
                TestOutcome::Falsified => "FAIL",
                TestOutcome::Skipped => "SKIP",
                TestOutcome::Error => "ERR",
            };
            writeln!(
                out,
                "[{}] {}: {} ({} pts)",
                status, result.id, result.name, result.points
            )
            .unwrap();
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let report = FalsificationReport::new("test-spec".to_string());
        assert_eq!(report.spec_name, "test-spec");
    }

    #[test]
    fn test_report_finalize() {
        let mut report = FalsificationReport::new("test".to_string());

        report.add_result(TestResult {
            id: "BC-001".to_string(),
            name: "Test 1".to_string(),
            category: "boundary".to_string(),
            points: 5,
            outcome: TestOutcome::Passed,
            error: None,
            evidence: vec![],
        });

        report.add_result(TestResult {
            id: "BC-002".to_string(),
            name: "Test 2".to_string(),
            category: "boundary".to_string(),
            points: 5,
            outcome: TestOutcome::Falsified,
            error: Some("Found edge case".to_string()),
            evidence: vec!["Input: empty".to_string()],
        });

        report.finalize();

        assert_eq!(report.summary.total_tests, 2);
        assert_eq!(report.summary.passed, 1);
        assert_eq!(report.summary.falsified, 1);
        assert!((report.summary.falsification_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_format_markdown() {
        let mut report = FalsificationReport::new("test".to_string());
        report.add_result(TestResult {
            id: "BC-001".to_string(),
            name: "Empty input".to_string(),
            category: "boundary".to_string(),
            points: 5,
            outcome: TestOutcome::Passed,
            error: None,
            evidence: vec![],
        });
        report.finalize();

        let md = report.format_markdown();
        assert!(md.contains("Falsification Report"));
        assert!(md.contains("boundary"));
    }
}
