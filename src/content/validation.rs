//! Validation (Jidoka)
//!
//! Content validation for quality gates with stop-on-error behavior.

use super::ContentType;
use serde::{Deserialize, Serialize};

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical - must halt
    Critical,
    /// Error - should halt
    Error,
    /// Warning - flag for revision
    Warning,
    /// Info - informational
    Info,
}

/// A single validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Constraint that was violated
    pub constraint: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Location in content (e.g., "paragraph 3", "code block 5")
    pub location: String,
    /// The offending text
    pub text: String,
    /// Suggested fix
    pub suggestion: String,
}

impl ValidationViolation {
    pub(crate) fn new(
        constraint: &str,
        severity: ValidationSeverity,
        location: String,
        text: String,
        suggestion: &str,
    ) -> Self {
        Self {
            constraint: constraint.to_string(),
            severity,
            location,
            text,
            suggestion: suggestion.to_string(),
        }
    }
}

/// Validation result from content validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Quality score (0-100)
    pub score: u8,
    /// List of violations
    pub violations: Vec<ValidationViolation>,
}

impl ValidationResult {
    /// Create a passing result
    pub fn pass(score: u8) -> Self {
        Self {
            passed: true,
            score,
            violations: Vec::new(),
        }
    }

    /// Create a failing result
    pub fn fail(violations: Vec<ValidationViolation>) -> Self {
        let score = Self::calculate_score(&violations);
        Self {
            passed: false,
            score,
            violations,
        }
    }

    /// Add a violation
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
        self.score = Self::calculate_score(&self.violations);
        self.passed = !self.violations.iter().any(|v| {
            matches!(
                v.severity,
                ValidationSeverity::Critical | ValidationSeverity::Error
            )
        });
    }

    /// Calculate score based on violations
    fn calculate_score(violations: &[ValidationViolation]) -> u8 {
        let mut score = 100i32;
        for v in violations {
            match v.severity {
                ValidationSeverity::Critical => score -= 50,
                ValidationSeverity::Error => score -= 25,
                ValidationSeverity::Warning => score -= 10,
                ValidationSeverity::Info => score -= 2,
            }
        }
        score.max(0) as u8
    }

    /// Check if there are critical violations
    pub fn has_critical(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ValidationSeverity::Critical)
    }

    /// Check if there are errors
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ValidationSeverity::Error)
    }

    /// Format as display string
    pub fn format_display(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Quality Score: {}/100\n\n", self.score));

        if self.violations.is_empty() {
            output.push_str("No violations found. ✓\n");
            return output;
        }

        output.push_str(&format!("Violations ({}):\n", self.violations.len()));
        for (i, v) in self.violations.iter().enumerate() {
            let prefix = if i == self.violations.len() - 1 {
                "└──"
            } else {
                "├──"
            };
            let severity = match v.severity {
                ValidationSeverity::Critical => "CRITICAL",
                ValidationSeverity::Error => "ERROR",
                ValidationSeverity::Warning => "WARNING",
                ValidationSeverity::Info => "INFO",
            };
            output.push_str(&format!(
                "{} [{}] {} @ {}\n",
                prefix, severity, v.constraint, v.location
            ));
            output.push_str(&format!("    Text: \"{}\"\n", v.text));
            output.push_str(&format!("    Fix: {}\n", v.suggestion));
        }

        output
    }
}

/// Content validator for Jidoka quality gates
#[derive(Debug, Clone)]
pub struct ContentValidator {
    /// Content type being validated
    content_type: ContentType,
}

impl ContentValidator {
    /// Create a new validator for a content type
    pub fn new(content_type: ContentType) -> Self {
        Self { content_type }
    }

    /// Validate content against all rules
    pub fn validate(&self, content: &str) -> ValidationResult {
        let mut result = ValidationResult::pass(100);

        // Run all validation checks
        self.validate_instructor_voice(content, &mut result);
        self.validate_code_blocks(content, &mut result);
        self.validate_heading_hierarchy(content, &mut result);
        self.validate_meta_commentary(content, &mut result);

        // Content-type specific validation
        match self.content_type {
            ContentType::BookChapter | ContentType::BlogPost => {
                self.validate_frontmatter(content, &mut result);
            }
            _ => {}
        }

        result
    }

    /// Check for meta-commentary (Andon)
    fn validate_meta_commentary(&self, content: &str, result: &mut ValidationResult) {
        let meta_phrases = [
            "in this chapter",
            "in this section",
            "we will learn",
            "we will explore",
            "we will discuss",
            "this chapter covers",
            "this section covers",
            "as mentioned earlier",
            "as we discussed",
        ];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            for phrase in &meta_phrases {
                if lower.contains(phrase) {
                    result.add_violation(ValidationViolation::new(
                        "no_meta_commentary",
                        ValidationSeverity::Warning,
                        format!("line {}", line_num + 1),
                        line.trim().chars().take(60).collect::<String>() + "...",
                        "Use direct instruction instead of meta-commentary",
                    ));
                }
            }
        }
    }

    /// Validate instructor voice
    fn validate_instructor_voice(&self, content: &str, result: &mut ValidationResult) {
        // Check for passive voice indicators in instruction contexts
        let passive_indicators = [
            "is being",
            "was being",
            "has been",
            "have been",
            "will be shown",
            "can be seen",
        ];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            // Only check non-code lines
            if !line.trim().starts_with("```") && !line.trim().starts_with("//") {
                for phrase in &passive_indicators {
                    if lower.contains(phrase) {
                        result.add_violation(ValidationViolation::new(
                            "instructor_voice",
                            ValidationSeverity::Info,
                            format!("line {}", line_num + 1),
                            line.trim().chars().take(60).collect::<String>(),
                            "Consider using active voice for clearer instruction",
                        ));
                    }
                }
            }
        }
    }

    /// Validate code blocks have language specifiers
    fn validate_code_blocks(&self, content: &str, result: &mut ValidationResult) {
        let mut in_code_block = false;
        let mut block_start = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().starts_with("```") {
                if !in_code_block {
                    // Starting a code block
                    in_code_block = true;
                    block_start = line_num + 1;
                    let lang = line.trim().trim_start_matches('`');
                    if lang.is_empty() {
                        result.add_violation(ValidationViolation::new(
                            "code_block_language",
                            ValidationSeverity::Warning,
                            format!("line {}", line_num + 1),
                            "```".to_string(),
                            "Specify language: ```rust, ```python, ```bash, etc.",
                        ));
                    }
                } else {
                    // Ending a code block
                    in_code_block = false;
                }
            }
        }

        // Check for unclosed code block
        if in_code_block {
            result.add_violation(ValidationViolation::new(
                "code_block_closed",
                ValidationSeverity::Error,
                format!("line {}", block_start),
                "Unclosed code block".to_string(),
                "Add closing ``` to code block",
            ));
        }
    }

    /// Validate heading hierarchy (no skipped levels)
    fn validate_heading_hierarchy(&self, content: &str, result: &mut ValidationResult) {
        let mut last_level = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.starts_with('#') {
                let level = line.chars().take_while(|c| *c == '#').count();
                if last_level > 0 && level > last_level + 1 {
                    result.add_violation(ValidationViolation::new(
                        "heading_hierarchy",
                        ValidationSeverity::Error,
                        format!("line {}", line_num + 1),
                        line.trim().to_string(),
                        &format!(
                            "Heading level {} skips from level {}. Use H{}.",
                            level,
                            last_level,
                            last_level + 1
                        ),
                    ));
                }
                last_level = level;
            }
        }
    }

    /// Validate frontmatter presence (for BCH and BLP)
    fn validate_frontmatter(&self, content: &str, result: &mut ValidationResult) {
        let _has_yaml_frontmatter = content.starts_with("---");
        let has_toml_frontmatter = content.starts_with("+++");

        if self.content_type == ContentType::BlogPost && !has_toml_frontmatter {
            result.add_violation(ValidationViolation::new(
                "frontmatter_present",
                ValidationSeverity::Critical,
                "beginning".to_string(),
                "Missing TOML frontmatter".to_string(),
                "Add +++ frontmatter with title, date, description",
            ));
        }
    }
}
