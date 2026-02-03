//! Stack Compliance Rule trait
//!
//! Defines the interface for compliance rules that can be checked
//! across the Sovereign AI Stack projects.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// Result of a compliance rule check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleResult {
    /// Whether the rule passed
    pub passed: bool,
    /// List of violations found
    pub violations: Vec<RuleViolation>,
    /// Suggestions for improvement (not violations)
    pub suggestions: Vec<Suggestion>,
    /// Additional context/metadata
    pub context: Option<String>,
}

impl RuleResult {
    /// Create a passing result
    pub fn pass() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
            suggestions: Vec::new(),
            context: None,
        }
    }

    /// Create a passing result with suggestions
    pub fn pass_with_suggestions(suggestions: Vec<Suggestion>) -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
            suggestions,
            context: None,
        }
    }

    /// Create a failing result with violations
    pub fn fail(violations: Vec<RuleViolation>) -> Self {
        Self {
            passed: false,
            violations,
            suggestions: Vec::new(),
            context: None,
        }
    }

    /// Add context to the result
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// A specific violation of a compliance rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    /// Violation code (e.g., "MK-001")
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Severity of the violation
    pub severity: ViolationLevel,
    /// File or location where violation was found
    pub location: Option<String>,
    /// Line number (if applicable)
    pub line: Option<usize>,
    /// Expected value/content
    pub expected: Option<String>,
    /// Actual value/content found
    pub actual: Option<String>,
    /// Whether this violation is auto-fixable
    pub fixable: bool,
}

impl RuleViolation {
    /// Create a new violation
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity: ViolationLevel::Error,
            location: None,
            line: None,
            expected: None,
            actual: None,
            fixable: false,
        }
    }

    /// Set the severity
    pub fn with_severity(mut self, severity: ViolationLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Set the location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set the line number
    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    /// Set expected/actual values
    pub fn with_diff(
        mut self,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        self.expected = Some(expected.into());
        self.actual = Some(actual.into());
        self
    }

    /// Mark as fixable
    pub fn fixable(mut self) -> Self {
        self.fixable = true;
        self
    }
}

/// Severity level of a violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationLevel {
    /// Informational - not a real violation
    Info,
    /// Warning - should be fixed but not blocking
    Warning,
    /// Error - must be fixed
    Error,
    /// Critical - blocks releases
    Critical,
}

impl fmt::Display for ViolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationLevel::Info => write!(f, "INFO"),
            ViolationLevel::Warning => write!(f, "WARN"),
            ViolationLevel::Error => write!(f, "ERROR"),
            ViolationLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A suggestion for improvement (not a violation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// Suggestion message
    pub message: String,
    /// Location (if applicable)
    pub location: Option<String>,
    /// Suggested fix
    pub fix: Option<String>,
}

impl Suggestion {
    /// Create a new suggestion
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            location: None,
            fix: None,
        }
    }

    /// Add location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Add suggested fix
    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

/// Result of attempting to fix violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixResult {
    /// Whether all fixes were applied successfully
    pub success: bool,
    /// Number of violations fixed
    pub fixed_count: usize,
    /// Number of violations that couldn't be fixed
    pub failed_count: usize,
    /// Details about each fix attempt
    pub details: Vec<FixDetail>,
}

impl FixResult {
    /// Create a successful fix result
    pub fn success(fixed: usize) -> Self {
        Self {
            success: true,
            fixed_count: fixed,
            failed_count: 0,
            details: Vec::new(),
        }
    }

    /// Create a partial fix result
    pub fn partial(fixed: usize, failed: usize, details: Vec<FixDetail>) -> Self {
        Self {
            success: false,
            fixed_count: fixed,
            failed_count: failed,
            details,
        }
    }

    /// Create a failed fix result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            fixed_count: 0,
            failed_count: 0,
            details: vec![FixDetail::Error(error.into())],
        }
    }

    /// Add a detail
    pub fn with_detail(mut self, detail: FixDetail) -> Self {
        self.details.push(detail);
        self
    }
}

/// Detail about a specific fix attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixDetail {
    /// Successfully applied fix
    Fixed {
        code: String,
        description: String,
    },
    /// Failed to apply fix
    FailedToFix {
        code: String,
        reason: String,
    },
    /// General error
    Error(String),
}

/// Trait for stack compliance rules
///
/// Implement this trait to create custom compliance rules.
pub trait StackComplianceRule: Send + Sync + std::fmt::Debug {
    /// Unique identifier for this rule (e.g., "makefile-targets")
    fn id(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Detailed help text (optional)
    fn help(&self) -> Option<&str> {
        None
    }

    /// Check a project for compliance
    fn check(&self, project_path: &Path) -> anyhow::Result<RuleResult>;

    /// Whether this rule can auto-fix violations
    fn can_fix(&self) -> bool {
        false
    }

    /// Attempt to fix violations (if supported)
    fn fix(&self, project_path: &Path) -> anyhow::Result<FixResult> {
        let _ = project_path;
        Ok(FixResult::failure("Auto-fix not supported for this rule"))
    }

    /// Category of this rule
    fn category(&self) -> RuleCategory {
        RuleCategory::General
    }
}

/// Category of compliance rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleCategory {
    /// General project structure
    General,
    /// Build system (Makefile, Cargo)
    Build,
    /// CI/CD workflows
    Ci,
    /// Code quality
    Code,
    /// Documentation
    Docs,
}

impl fmt::Display for RuleCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleCategory::General => write!(f, "General"),
            RuleCategory::Build => write!(f, "Build"),
            RuleCategory::Ci => write!(f, "CI"),
            RuleCategory::Code => write!(f, "Code"),
            RuleCategory::Docs => write!(f, "Docs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_result_pass() {
        let result = RuleResult::pass();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_rule_result_fail() {
        let violations = vec![RuleViolation::new("TEST-001", "Test violation")];
        let result = RuleResult::fail(violations);
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_violation_builder() {
        let violation = RuleViolation::new("MK-001", "Missing target")
            .with_severity(ViolationLevel::Error)
            .with_location("Makefile")
            .with_line(10)
            .with_diff("test-fast", "test")
            .fixable();

        assert_eq!(violation.code, "MK-001");
        assert_eq!(violation.severity, ViolationLevel::Error);
        assert_eq!(violation.location, Some("Makefile".to_string()));
        assert_eq!(violation.line, Some(10));
        assert!(violation.fixable);
    }

    #[test]
    fn test_fix_result() {
        let result = FixResult::success(5);
        assert!(result.success);
        assert_eq!(result.fixed_count, 5);
    }

    #[test]
    fn test_violation_level_display() {
        assert_eq!(format!("{}", ViolationLevel::Info), "INFO");
        assert_eq!(format!("{}", ViolationLevel::Warning), "WARN");
        assert_eq!(format!("{}", ViolationLevel::Error), "ERROR");
        assert_eq!(format!("{}", ViolationLevel::Critical), "CRITICAL");
    }
}
