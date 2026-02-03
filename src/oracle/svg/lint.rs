//! SVG Linting and Validation
//!
//! Rules for validating SVG diagrams against Material Design 3 guidelines.

use super::layout::{LayoutEngine, LayoutError, GRID_SIZE};
use super::palette::{Color, MaterialPalette};

/// Lint severity level
/// Ordered from least to most severe for comparison purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintSeverity {
    /// Info - suggestion
    Info,
    /// Warning - should fix
    Warning,
    /// Error - must fix
    Error,
}

impl std::fmt::Display for LintSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A lint violation
#[derive(Debug, Clone)]
pub struct LintViolation {
    /// Rule that was violated
    pub rule: LintRule,
    /// Severity
    pub severity: LintSeverity,
    /// Human-readable message
    pub message: String,
    /// Element ID if applicable
    pub element_id: Option<String>,
}

impl std::fmt::Display for LintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(id) = &self.element_id {
            write!(f, "[{}] {}: {} (element: {})", self.severity, self.rule, self.message, id)
        } else {
            write!(f, "[{}] {}: {}", self.severity, self.rule, self.message)
        }
    }
}

/// Lint rule identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintRule {
    /// No overlapping elements on same layer
    NoOverlap,
    /// All colors from Material palette
    MaterialColors,
    /// 8px grid alignment
    GridAlignment,
    /// File size under 100KB
    FileSize,
    /// Elements within viewport
    WithinBounds,
    /// Minimum contrast ratio
    ContrastRatio,
    /// Consistent stroke widths
    StrokeConsistency,
    /// Text minimum size
    MinTextSize,
}

impl std::fmt::Display for LintRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOverlap => write!(f, "NO_OVERLAP"),
            Self::MaterialColors => write!(f, "MATERIAL_COLORS"),
            Self::GridAlignment => write!(f, "GRID_ALIGNMENT"),
            Self::FileSize => write!(f, "FILE_SIZE"),
            Self::WithinBounds => write!(f, "WITHIN_BOUNDS"),
            Self::ContrastRatio => write!(f, "CONTRAST_RATIO"),
            Self::StrokeConsistency => write!(f, "STROKE_CONSISTENCY"),
            Self::MinTextSize => write!(f, "MIN_TEXT_SIZE"),
        }
    }
}

/// SVG linter configuration
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Maximum file size in bytes
    pub max_file_size: usize,
    /// Grid size for alignment checks
    pub grid_size: f32,
    /// Minimum text size in pixels
    pub min_text_size: f32,
    /// Minimum contrast ratio (WCAG AA is 4.5:1)
    pub min_contrast_ratio: f32,
    /// Check material colors
    pub check_material_colors: bool,
    /// Check grid alignment
    pub check_grid_alignment: bool,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100_000, // 100KB
            grid_size: GRID_SIZE,
            min_text_size: 11.0, // Label small
            min_contrast_ratio: 4.5, // WCAG AA
            check_material_colors: true,
            check_grid_alignment: true,
        }
    }
}

/// SVG Linter
#[derive(Debug)]
pub struct SvgLinter {
    config: LintConfig,
    palette: MaterialPalette,
}

impl SvgLinter {
    /// Create a new linter with default config
    pub fn new() -> Self {
        Self {
            config: LintConfig::default(),
            palette: MaterialPalette::light(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: LintConfig) -> Self {
        Self {
            config,
            palette: MaterialPalette::light(),
        }
    }

    /// Set the palette to validate against
    pub fn with_palette(mut self, palette: MaterialPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Lint the layout engine for overlap and bounds issues
    pub fn lint_layout(&self, layout: &LayoutEngine) -> Vec<LintViolation> {
        let mut violations = Vec::new();

        for error in layout.validate() {
            let violation = match error {
                LayoutError::Overlap { id1, id2 } => LintViolation {
                    rule: LintRule::NoOverlap,
                    severity: LintSeverity::Error,
                    message: format!("Elements '{}' and '{}' overlap", id1, id2),
                    element_id: Some(id1),
                },
                LayoutError::OutOfBounds { id } => LintViolation {
                    rule: LintRule::WithinBounds,
                    severity: LintSeverity::Error,
                    message: "Element is outside viewport bounds".to_string(),
                    element_id: Some(id),
                },
                LayoutError::NotAligned { id } => {
                    if self.config.check_grid_alignment {
                        LintViolation {
                            rule: LintRule::GridAlignment,
                            severity: LintSeverity::Warning,
                            message: format!("Element is not aligned to {}px grid", self.config.grid_size),
                            element_id: Some(id),
                        }
                    } else {
                        continue;
                    }
                }
            };
            violations.push(violation);
        }

        violations
    }

    /// Check if a color is valid (in the material palette)
    pub fn lint_color(&self, color: &Color, element_id: Option<&str>) -> Option<LintViolation> {
        if !self.config.check_material_colors {
            return None;
        }

        if !self.palette.is_valid_color(color) {
            Some(LintViolation {
                rule: LintRule::MaterialColors,
                severity: LintSeverity::Warning,
                message: format!("Color {} is not in the Material palette", color.to_css_hex()),
                element_id: element_id.map(|s| s.to_string()),
            })
        } else {
            None
        }
    }

    /// Check file size
    pub fn lint_file_size(&self, svg_content: &str) -> Option<LintViolation> {
        if svg_content.len() > self.config.max_file_size {
            Some(LintViolation {
                rule: LintRule::FileSize,
                severity: LintSeverity::Error,
                message: format!(
                    "File size {} bytes exceeds maximum {} bytes",
                    svg_content.len(),
                    self.config.max_file_size
                ),
                element_id: None,
            })
        } else {
            None
        }
    }

    /// Check text size
    pub fn lint_text_size(&self, size: f32, element_id: Option<&str>) -> Option<LintViolation> {
        if size < self.config.min_text_size {
            Some(LintViolation {
                rule: LintRule::MinTextSize,
                severity: LintSeverity::Warning,
                message: format!(
                    "Text size {}px is below minimum {}px",
                    size, self.config.min_text_size
                ),
                element_id: element_id.map(|s| s.to_string()),
            })
        } else {
            None
        }
    }

    /// Calculate luminance for contrast ratio
    fn relative_luminance(color: &Color) -> f64 {
        fn channel_luminance(c: u8) -> f64 {
            let c = c as f64 / 255.0;
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }

        let r = channel_luminance(color.r);
        let g = channel_luminance(color.g);
        let b = channel_luminance(color.b);

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Calculate contrast ratio between two colors
    pub fn contrast_ratio(color1: &Color, color2: &Color) -> f64 {
        let l1 = Self::relative_luminance(color1);
        let l2 = Self::relative_luminance(color2);

        let lighter = l1.max(l2);
        let darker = l1.min(l2);

        (lighter + 0.05) / (darker + 0.05)
    }

    /// Check contrast ratio between foreground and background
    pub fn lint_contrast(
        &self,
        foreground: &Color,
        background: &Color,
        element_id: Option<&str>,
    ) -> Option<LintViolation> {
        let ratio = Self::contrast_ratio(foreground, background);

        if ratio < self.config.min_contrast_ratio as f64 {
            Some(LintViolation {
                rule: LintRule::ContrastRatio,
                severity: LintSeverity::Warning,
                message: format!(
                    "Contrast ratio {:.2}:1 is below minimum {:.1}:1 (WCAG AA)",
                    ratio, self.config.min_contrast_ratio
                ),
                element_id: element_id.map(|s| s.to_string()),
            })
        } else {
            None
        }
    }

    /// Run all lint checks and return violations
    pub fn lint_all(
        &self,
        layout: &LayoutEngine,
        svg_content: &str,
        colors: &[(&str, Color)],
        text_sizes: &[(&str, f32)],
    ) -> LintResult {
        let mut violations = Vec::new();

        // Layout checks
        violations.extend(self.lint_layout(layout));

        // File size check
        if let Some(v) = self.lint_file_size(svg_content) {
            violations.push(v);
        }

        // Color checks
        for (id, color) in colors {
            if let Some(v) = self.lint_color(color, Some(id)) {
                violations.push(v);
            }
        }

        // Text size checks
        for (id, size) in text_sizes {
            if let Some(v) = self.lint_text_size(*size, Some(id)) {
                violations.push(v);
            }
        }

        LintResult::new(violations)
    }
}

impl Default for SvgLinter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of linting
#[derive(Debug)]
pub struct LintResult {
    /// All violations found
    pub violations: Vec<LintViolation>,
}

impl LintResult {
    /// Create a new lint result
    pub fn new(violations: Vec<LintViolation>) -> Self {
        Self { violations }
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == LintSeverity::Error)
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == LintSeverity::Warning)
    }

    /// Check if lint passed (no errors)
    pub fn passed(&self) -> bool {
        !self.has_errors()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == LintSeverity::Error)
            .count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == LintSeverity::Warning)
            .count()
    }

    /// Get violations by severity
    pub fn by_severity(&self, severity: LintSeverity) -> Vec<&LintViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity == severity)
            .collect()
    }

    /// Get violations by rule
    pub fn by_rule(&self, rule: LintRule) -> Vec<&LintViolation> {
        self.violations.iter().filter(|v| v.rule == rule).collect()
    }
}

impl std::fmt::Display for LintResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.violations.is_empty() {
            return writeln!(f, "Lint passed: no violations");
        }

        writeln!(
            f,
            "Lint result: {} error(s), {} warning(s)",
            self.error_count(),
            self.warning_count()
        )?;

        for violation in &self.violations {
            writeln!(f, "  {}", violation)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::svg::layout::Viewport;
    use crate::oracle::svg::shapes::Rect;

    #[test]
    fn test_lint_severity_order() {
        assert!(LintSeverity::Error > LintSeverity::Warning);
        assert!(LintSeverity::Warning > LintSeverity::Info);
    }

    #[test]
    fn test_linter_creation() {
        let linter = SvgLinter::new();
        assert_eq!(linter.config.max_file_size, 100_000);
        assert_eq!(linter.config.grid_size, 8.0);
    }

    #[test]
    fn test_lint_color_valid() {
        let linter = SvgLinter::new();
        let palette = MaterialPalette::light();

        // Valid color
        let violation = linter.lint_color(&palette.primary, Some("test"));
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_color_invalid() {
        let linter = SvgLinter::new();

        // Invalid color (not in palette)
        let violation = linter.lint_color(&Color::rgb(1, 2, 3), Some("test"));
        assert!(violation.is_some());
        assert_eq!(violation.unwrap().rule, LintRule::MaterialColors);
    }

    #[test]
    fn test_lint_file_size_ok() {
        let linter = SvgLinter::new();
        let small_svg = "<svg></svg>";

        let violation = linter.lint_file_size(small_svg);
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_file_size_too_large() {
        let config = LintConfig {
            max_file_size: 10,
            ..Default::default()
        };
        let linter = SvgLinter::with_config(config);

        let violation = linter.lint_file_size("This is longer than 10 bytes");
        assert!(violation.is_some());
        assert_eq!(violation.unwrap().rule, LintRule::FileSize);
    }

    #[test]
    fn test_lint_text_size() {
        let linter = SvgLinter::new();

        // Too small
        let violation = linter.lint_text_size(8.0, Some("text1"));
        assert!(violation.is_some());
        assert_eq!(violation.unwrap().rule, LintRule::MinTextSize);

        // OK
        let violation = linter.lint_text_size(14.0, Some("text2"));
        assert!(violation.is_none());
    }

    #[test]
    fn test_contrast_ratio_calculation() {
        // Black on white should be ~21:1
        let ratio = SvgLinter::contrast_ratio(&Color::rgb(0, 0, 0), &Color::rgb(255, 255, 255));
        assert!(ratio > 20.0 && ratio < 22.0);

        // Same colors should be 1:1
        let ratio = SvgLinter::contrast_ratio(&Color::rgb(128, 128, 128), &Color::rgb(128, 128, 128));
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lint_contrast() {
        let linter = SvgLinter::new();

        // Good contrast (black on white)
        let violation = linter.lint_contrast(
            &Color::rgb(0, 0, 0),
            &Color::rgb(255, 255, 255),
            Some("text"),
        );
        assert!(violation.is_none());

        // Poor contrast (light gray on white)
        let violation = linter.lint_contrast(
            &Color::rgb(200, 200, 200),
            &Color::rgb(255, 255, 255),
            Some("text"),
        );
        assert!(violation.is_some());
    }

    #[test]
    fn test_lint_layout_overlap() {
        let mut layout = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));

        // Add overlapping elements directly to test validation
        layout
            .elements
            .insert("r1".to_string(), super::super::layout::LayoutRect::new("r1", Rect::new(0.0, 0.0, 50.0, 50.0)));
        layout
            .elements
            .insert("r2".to_string(), super::super::layout::LayoutRect::new("r2", Rect::new(25.0, 25.0, 50.0, 50.0)));

        let linter = SvgLinter::new();
        let violations = linter.lint_layout(&layout);

        assert!(violations.iter().any(|v| v.rule == LintRule::NoOverlap));
    }

    #[test]
    fn test_lint_result() {
        let violations = vec![
            LintViolation {
                rule: LintRule::NoOverlap,
                severity: LintSeverity::Error,
                message: "Overlap".to_string(),
                element_id: Some("r1".to_string()),
            },
            LintViolation {
                rule: LintRule::MaterialColors,
                severity: LintSeverity::Warning,
                message: "Bad color".to_string(),
                element_id: Some("r2".to_string()),
            },
        ];

        let result = LintResult::new(violations);

        assert!(result.has_errors());
        assert!(result.has_warnings());
        assert!(!result.passed());
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.warning_count(), 1);
    }

    #[test]
    fn test_lint_result_passed() {
        let result = LintResult::new(vec![]);
        assert!(result.passed());
        assert!(!result.has_errors());
    }

    #[test]
    fn test_lint_result_display() {
        let result = LintResult::new(vec![]);
        let output = format!("{}", result);
        assert!(output.contains("no violations"));
    }
}
