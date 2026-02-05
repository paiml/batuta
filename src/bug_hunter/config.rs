//! Bug Hunter Configuration
//!
//! Handles loading and parsing of `.pmat/bug-hunter.toml` configuration files.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Bug Hunter configuration loaded from `.pmat/bug-hunter.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BugHunterConfig {
    /// Allowlist entries for intentional patterns
    #[serde(default)]
    pub allow: Vec<AllowEntry>,

    /// Custom pattern definitions
    #[serde(default)]
    pub patterns: Vec<CustomPattern>,

    /// Trend tracking settings
    #[serde(default)]
    pub trend: TrendConfig,
}

/// An allowlist entry marking a pattern as intentional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowEntry {
    /// File glob pattern (e.g., "src/optim/*.rs")
    pub file: String,

    /// Pattern to allow (e.g., "unimplemented")
    pub pattern: String,

    /// Reason for allowing (documentation)
    #[serde(default)]
    pub reason: String,

    /// Optional: only allow in specific line ranges
    #[serde(default)]
    pub lines: Option<LineRange>,
}

/// Line range for scoped allowlist entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
}

/// A custom pattern definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    /// The pattern to match (regex or literal)
    pub pattern: String,

    /// Category for the finding
    #[serde(default = "default_category")]
    pub category: String,

    /// Severity level
    #[serde(default = "default_severity")]
    pub severity: String,

    /// Suspiciousness score (0.0-1.0)
    #[serde(default = "default_suspiciousness")]
    pub suspiciousness: f64,

    /// Optional description
    #[serde(default)]
    pub description: String,

    /// File glob to limit scope (optional)
    #[serde(default)]
    pub file_glob: Option<String>,

    /// Language filter (optional: "rust", "python", "typescript", "go")
    #[serde(default)]
    pub language: Option<String>,
}

fn default_category() -> String {
    "Custom".to_string()
}

fn default_severity() -> String {
    "Medium".to_string()
}

fn default_suspiciousness() -> f64 {
    0.5
}

/// Trend tracking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendConfig {
    /// Enable automatic trend snapshots
    #[serde(default)]
    pub enabled: bool,

    /// Snapshot interval in days
    #[serde(default = "default_interval")]
    pub interval_days: u32,

    /// Maximum snapshots to retain
    #[serde(default = "default_max_snapshots")]
    pub max_snapshots: usize,
}

fn default_interval() -> u32 {
    7
}

fn default_max_snapshots() -> usize {
    52
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_days: default_interval(),
            max_snapshots: default_max_snapshots(),
        }
    }
}

impl BugHunterConfig {
    /// Load configuration from a project path.
    ///
    /// Looks for `.pmat/bug-hunter.toml` in the project root.
    pub fn load(project_path: &Path) -> Self {
        let config_path = project_path.join(".pmat").join("bug-hunter.toml");
        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => return config,
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to parse {}: {}",
                            config_path.display(),
                            e
                        );
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to read {}: {}",
                        config_path.display(),
                        e
                    );
                }
            }
        }
        Self::default()
    }

    /// Check if a finding should be allowed (skipped).
    pub fn is_allowed(&self, file_path: &Path, pattern: &str, line: usize) -> bool {
        let file_str = file_path.to_string_lossy();

        for entry in &self.allow {
            // Check pattern match
            if !entry.pattern.eq_ignore_ascii_case(pattern) && entry.pattern != "*" {
                continue;
            }

            // Check file glob
            if !glob_match(&entry.file, &file_str) {
                continue;
            }

            // Check line range if specified
            if let Some(ref range) = entry.lines {
                if line < range.start || line > range.end {
                    continue;
                }
            }

            return true;
        }

        false
    }
}

/// Simple glob matching (supports * and **).
fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern == "*" || pattern == "**" {
        return true;
    }

    // Convert glob to regex-like matching
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    glob_match_parts(&pattern_parts, &path_parts)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }

    let p = pattern[0];

    if p == "**" {
        // ** matches zero or more path segments
        for i in 0..=path.len() {
            if glob_match_parts(&pattern[1..], &path[i..]) {
                return true;
            }
        }
        return false;
    }

    if path.is_empty() {
        return false;
    }

    // Check if segment matches (with * wildcard)
    if segment_matches(p, path[0]) {
        glob_match_parts(&pattern[1..], &path[1..])
    } else {
        false
    }
}

fn segment_matches(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if !pattern.contains('*') {
        return pattern == segment;
    }

    // Simple wildcard matching
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 2 {
        let (prefix, suffix) = (parts[0], parts[1]);
        return segment.starts_with(prefix) && segment.ends_with(suffix);
    }

    // Fallback to exact match for complex patterns
    pattern == segment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match_simple() {
        assert!(glob_match("src/*.rs", "src/main.rs"));
        assert!(glob_match("src/*.rs", "src/lib.rs"));
        assert!(!glob_match("src/*.rs", "src/foo/bar.rs"));
    }

    #[test]
    fn test_glob_match_double_star() {
        assert!(glob_match("src/**/*.rs", "src/main.rs"));
        assert!(glob_match("src/**/*.rs", "src/foo/bar.rs"));
        assert!(glob_match("src/**/*.rs", "src/foo/bar/baz.rs"));
        assert!(!glob_match("src/**/*.rs", "test/main.rs"));
    }

    #[test]
    fn test_glob_match_star() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("**", "any/path/here"));
    }

    #[test]
    fn test_is_allowed() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/optim/*.rs".to_string(),
                pattern: "unimplemented".to_string(),
                reason: "Batch optimizers don't support step()".to_string(),
                lines: None,
            }],
            ..Default::default()
        };

        assert!(config.is_allowed(Path::new("src/optim/admm.rs"), "unimplemented", 100));
        assert!(!config.is_allowed(Path::new("src/main.rs"), "unimplemented", 100));
        assert!(!config.is_allowed(Path::new("src/optim/admm.rs"), "placeholder", 100));
    }

    #[test]
    fn test_is_allowed_with_line_range() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/foo.rs".to_string(),
                pattern: "TODO".to_string(),
                reason: "Known issue".to_string(),
                lines: Some(LineRange { start: 10, end: 20 }),
            }],
            ..Default::default()
        };

        assert!(config.is_allowed(Path::new("src/foo.rs"), "TODO", 15));
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 5));
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 25));
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
[[allow]]
file = "src/optim/*.rs"
pattern = "unimplemented"
reason = "Batch optimizers"

[[patterns]]
pattern = "PERF-TODO"
category = "PerformanceDebt"
severity = "High"
suspiciousness = 0.8
"#;

        let config: BugHunterConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.allow.len(), 1);
        assert_eq!(config.patterns.len(), 1);
        assert_eq!(config.patterns[0].pattern, "PERF-TODO");
        assert_eq!(config.patterns[0].suspiciousness, 0.8);
    }
}
