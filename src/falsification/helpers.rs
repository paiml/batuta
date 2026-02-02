//! Shared helper functions for falsification checks.
//!
//! Eliminates duplicated pattern-matching logic across falsification modules.

use std::path::Path;

use super::types::CheckItem;

/// Represents the outcome to apply to a [`CheckItem`].
///
/// Used with [`apply_check_outcome`] to replace repeated if-else outcome chains
/// in falsification check functions.
pub(crate) enum CheckOutcome<'a> {
    /// Pass the check (no message).
    Pass,
    /// Partial pass with a diagnostic message.
    Partial(&'a str),
    /// Fail with a diagnostic message.
    Fail(&'a str),
}

/// Apply the first matching outcome to a check item.
///
/// Iterates through `checks` and applies the outcome for the first entry
/// whose condition is `true`. If no condition matches, returns the item unchanged.
pub(crate) fn apply_check_outcome(item: CheckItem, checks: &[(bool, CheckOutcome<'_>)]) -> CheckItem {
    for (condition, outcome) in checks {
        if *condition {
            return match outcome {
                CheckOutcome::Pass => item.pass(),
                CheckOutcome::Partial(msg) => item.partial(*msg),
                CheckOutcome::Fail(msg) => item.fail(*msg),
            };
        }
    }
    item
}

/// Check if any file matching the glob patterns contains any of the search patterns.
///
/// Used by all falsification modules for source/config file scanning.
pub(crate) fn files_contain_pattern(
    project_path: &Path,
    glob_patterns: &[&str],
    search_patterns: &[&str],
) -> bool {
    for glob_pat in glob_patterns {
        let full = format!("{}/{}", project_path.display(), glob_pat);
        let Ok(entries) = glob::glob(&full) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            if search_patterns.iter().any(|p| content.contains(p)) {
                return true;
            }
        }
    }
    false
}

/// Check if any file matching the glob patterns contains any search pattern (case-insensitive).
pub(crate) fn files_contain_pattern_ci(
    project_path: &Path,
    glob_patterns: &[&str],
    search_patterns: &[&str],
) -> bool {
    for glob_pat in glob_patterns {
        let full = format!("{}/{}", project_path.display(), glob_pat);
        let Ok(entries) = glob::glob(&full) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let lower = content.to_lowercase();
            if search_patterns
                .iter()
                .any(|p| lower.contains(&p.to_lowercase()))
            {
                return true;
            }
        }
    }
    false
}

/// Check if any Rust source file contains any of the given patterns.
pub(crate) fn source_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern(project_path, &["src/**/*.rs"], patterns)
}

/// Check if any Rust source or config file contains any of the given patterns.
pub(crate) fn source_or_config_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern(
        project_path,
        &[
            "src/**/*.rs",
            "**/*.yaml",
            "**/*.toml",
            "**/*.json",
        ],
        patterns,
    )
}

/// Check if any CI workflow file contains any of the given patterns (case-insensitive).
pub(crate) fn ci_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern_ci(
        project_path,
        &[
            ".github/workflows/*.yml",
            ".github/workflows/*.yaml",
            ".gitlab-ci.yml",
        ],
        patterns,
    )
}

/// Check if any test file contains any of the given patterns.
///
/// Searches test directories, test-named files, and `#[cfg(test)]` modules.
pub(crate) fn test_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    if files_contain_pattern(
        project_path,
        &["tests/**/*.rs", "src/**/*test*.rs"],
        patterns,
    ) {
        return true;
    }

    // Also check #[cfg(test)] modules in source files
    let glob_str = format!("{}/src/**/*.rs", project_path.display());
    let Ok(entries) = glob::glob(&glob_str) else {
        return false;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("#[cfg(test)]") && patterns.iter().any(|p| content.contains(p)) {
            return true;
        }
    }
    false
}

/// Count how many platforms from the list appear in CI workflow files.
pub(crate) fn ci_platform_count(project_path: &Path, platforms: &[&str]) -> usize {
    let ci_globs = [
        format!("{}/.github/workflows/*.yml", project_path.display()),
        format!("{}/.github/workflows/*.yaml", project_path.display()),
    ];

    for glob_pattern in &ci_globs {
        let Ok(entries) = glob::glob(glob_pattern) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let count = platforms.iter().filter(|p| content.contains(*p)).count();
            if count >= 2 {
                return count;
            }
        }
    }
    0
}

/// Scan Cargo.toml for scripting runtime dependencies.
///
/// Returns list of forbidden dependency names found in non-dev deps.
pub(crate) fn find_scripting_deps(project_path: &Path) -> Vec<String> {
    let cargo_toml = project_path.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(&cargo_toml) else {
        return Vec::new();
    };

    let forbidden = ["pyo3", "napi", "mlua", "rlua", "rustpython"];
    let mut found = Vec::new();

    for dep in forbidden {
        let has_dep =
            content.contains(&format!("{dep} =")) || content.contains(&format!("{dep}="));
        if !has_dep {
            continue;
        }
        // Rough check: in [dependencies] but not after [dev-dependencies]
        let is_dev_only = content.contains("[dev-dependencies]")
            && content.find(&format!("{dep} =")) >= content.find("[dev-dependencies]");
        if !is_dev_only {
            found.push(dep.to_string());
        }
    }
    found
}

/// Detect serde/schema validation configuration from Cargo.toml.
pub(crate) struct SchemaInfo {
    pub has_serde: bool,
    pub has_serde_yaml: bool,
    pub has_validator: bool,
}

/// Read schema-related dependency info from Cargo.toml.
pub(crate) fn detect_schema_deps(project_path: &Path) -> SchemaInfo {
    let cargo_toml = project_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).unwrap_or_default();
    SchemaInfo {
        has_serde: content.contains("serde"),
        has_serde_yaml: content.contains("serde_yaml") || content.contains("serde_yml"),
        has_validator: content.contains("validator") || content.contains("garde"),
    }
}

/// Check if any source file has a config struct with Deserialize derive.
pub(crate) fn has_deserialize_config_struct(project_path: &Path) -> bool {
    let glob_str = format!("{}/src/**/*.rs", project_path.display());
    let Ok(entries) = glob::glob(&glob_str) else {
        return false;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("#[derive")
            && content.contains("Deserialize")
            && content.contains("struct")
            && content.to_lowercase().contains("config")
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_source_contains_pattern_finds_struct() {
        let path = PathBuf::from(".");
        assert!(source_contains_pattern(&path, &["struct"]));
    }

    #[test]
    fn test_source_contains_pattern_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!source_contains_pattern(&path, &["anything"]));
    }

    #[test]
    fn test_files_contain_pattern_ci_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!files_contain_pattern_ci(
            &path,
            &["src/**/*.rs"],
            &["anything"]
        ));
    }

    #[test]
    fn test_find_scripting_deps_current_project() {
        let path = PathBuf::from(".");
        let deps = find_scripting_deps(&path);
        // batuta should not have scripting deps
        assert!(deps.is_empty());
    }

    #[test]
    fn test_detect_schema_deps_current_project() {
        let path = PathBuf::from(".");
        let info = detect_schema_deps(&path);
        // batuta uses serde
        assert!(info.has_serde);
    }

    #[test]
    fn test_has_deserialize_config_struct_nonexistent() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!has_deserialize_config_struct(&path));
    }

    #[test]
    fn test_test_contains_pattern_finds_test() {
        let path = PathBuf::from(".");
        assert!(test_contains_pattern(&path, &["#[test]"]));
    }

    #[test]
    fn test_ci_platform_count_current_project() {
        let path = PathBuf::from(".");
        // May or may not have CI - just verify no panic
        let _ = ci_platform_count(&path, &["ubuntu", "macos", "windows"]);
    }
}
