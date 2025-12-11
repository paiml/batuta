//! Audit Utilities for Falsification Checks
//!
//! Provides reusable audit functions for the checklist evaluation.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of a file audit.
#[derive(Debug, Clone)]
pub struct FileAuditResult {
    /// Files matching the pattern
    pub matches: Vec<PathBuf>,
    /// Total files scanned
    pub scanned: usize,
    /// Patterns used
    pub patterns: Vec<String>,
}

impl FileAuditResult {
    /// Check if any violations were found.
    pub fn has_violations(&self) -> bool {
        !self.matches.is_empty()
    }

    /// Get violation count.
    pub fn violation_count(&self) -> usize {
        self.matches.len()
    }
}

/// Audit for scripting language files.
pub fn audit_scripting_files(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        "**/*.py".to_string(),
        "**/*.js".to_string(),
        "**/*.ts".to_string(),
        "**/*.lua".to_string(),
        "**/*.rb".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                scanned += 1;
                let path_str = entry.to_string_lossy();

                // Exclude common non-production paths
                let is_excluded = path_str.contains("node_modules")
                    || path_str.contains("venv")
                    || path_str.contains(".venv")
                    || path_str.contains("__pycache__")
                    || path_str.contains("/target/")
                    || path_str.contains("/dist/")
                    // Exclude example input files (for transpilation demos)
                    || path_str.contains("/examples/")
                    || path_str.contains("/migrations/")
                    // Exclude book/docs (mdBook generates JS)
                    || path_str.contains("/book/")
                    || path_str.contains("/docs/")
                    // Exclude test fixtures
                    || path_str.contains("/fixtures/")
                    || path_str.contains("/testdata/");

                if !is_excluded {
                    matches.push(entry);
                }
            }
        }
    }

    FileAuditResult {
        matches,
        scanned,
        patterns,
    }
}

/// Audit for test framework files.
pub fn audit_test_frameworks(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        // JavaScript test files
        "**/*.test.js".to_string(),
        "**/*.spec.js".to_string(),
        "**/*.test.ts".to_string(),
        "**/*.spec.ts".to_string(),
        "**/jest.config.*".to_string(),
        "**/vitest.config.*".to_string(),
        // Python test files
        "**/test_*.py".to_string(),
        "**/*_test.py".to_string(),
        "**/conftest.py".to_string(),
        "**/pytest.ini".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                scanned += 1;
                let path_str = entry.to_string_lossy();

                if !path_str.contains("node_modules") && !path_str.contains("venv") {
                    matches.push(entry);
                }
            }
        }
    }

    FileAuditResult {
        matches,
        scanned,
        patterns,
    }
}

/// Audit for YAML configuration files.
pub fn audit_yaml_configs(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        "*.yaml".to_string(),
        "*.yml".to_string(),
        "config/**/*.yaml".to_string(),
        "config/**/*.yml".to_string(),
        "examples/**/*.yaml".to_string(),
        "examples/**/*.yml".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                scanned += 1;
                matches.push(entry);
            }
        }
    }

    FileAuditResult {
        matches,
        scanned,
        patterns,
    }
}

/// Result of a dependency audit.
#[derive(Debug, Clone)]
pub struct DependencyAuditResult {
    /// Forbidden dependencies found
    pub forbidden: Vec<String>,
    /// All dependencies checked
    pub checked: Vec<String>,
    /// Raw cargo tree output (if available)
    pub cargo_tree_output: Option<String>,
}

impl DependencyAuditResult {
    /// Check if any forbidden dependencies were found.
    pub fn has_violations(&self) -> bool {
        !self.forbidden.is_empty()
    }
}

/// Audit Cargo.toml for forbidden dependencies.
pub fn audit_cargo_dependencies(project_path: &Path, forbidden: &[&str]) -> DependencyAuditResult {
    let cargo_toml = project_path.join("Cargo.toml");
    let mut found_forbidden = Vec::new();
    let mut checked = Vec::new();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            for dep in forbidden {
                checked.push(dep.to_string());

                // Simple check - a proper implementation would parse TOML
                if content.contains(&format!("{} =", dep))
                    || content.contains(&format!("{}=", dep))
                    || content.contains(&format!("\"{}\"", dep))
                {
                    found_forbidden.push(dep.to_string());
                }
            }
        }
    }

    // Try to get cargo tree output for more detailed analysis
    let cargo_tree_output = Command::new("cargo")
        .args(["tree", "--edges", "no-dev", "-p"])
        .current_dir(project_path)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        });

    DependencyAuditResult {
        forbidden: found_forbidden,
        checked,
        cargo_tree_output,
    }
}

/// Check if a Rust project has tests.
pub fn has_rust_tests(project_path: &Path) -> bool {
    // Check for tests directory
    if project_path.join("tests").exists() {
        return true;
    }

    // Check for #[test] or #[cfg(test)] in source files
    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                if content.contains("#[test]") || content.contains("#[cfg(test)]") {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if a project has WASM support.
pub fn has_wasm_support(project_path: &Path) -> WasmSupport {
    let cargo_toml = project_path.join("Cargo.toml");
    let mut support = WasmSupport::default();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            support.has_wasm_feature = content.contains("[features]")
                && (content.contains("wasm") || content.contains("web"));

            support.has_wasm_bindgen = content.contains("wasm-bindgen");
            support.has_web_sys = content.contains("web-sys");
            support.has_wasm_pack = content.contains("wasm-pack");
        }
    }

    // Check for wasm.rs module
    support.has_wasm_module = project_path.join("src/wasm.rs").exists();

    // Check for wasm target in .cargo/config.toml
    let cargo_config = project_path.join(".cargo/config.toml");
    if cargo_config.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_config) {
            support.has_wasm_target = content.contains("wasm32");
        }
    }

    support
}

/// WASM support detection result.
#[derive(Debug, Clone, Default)]
pub struct WasmSupport {
    pub has_wasm_feature: bool,
    pub has_wasm_bindgen: bool,
    pub has_web_sys: bool,
    pub has_wasm_pack: bool,
    pub has_wasm_module: bool,
    pub has_wasm_target: bool,
}

impl WasmSupport {
    /// Check if any WASM support is present.
    pub fn is_supported(&self) -> bool {
        self.has_wasm_feature
            || self.has_wasm_bindgen
            || self.has_web_sys
            || self.has_wasm_pack
            || self.has_wasm_module
    }

    /// Get support level description.
    pub fn level(&self) -> &'static str {
        if self.has_wasm_bindgen && self.has_wasm_module {
            "Full"
        } else if self.has_wasm_bindgen || self.has_wasm_feature {
            "Partial"
        } else if self.has_wasm_module {
            "Basic"
        } else {
            "None"
        }
    }
}

/// Check for serde-based config validation.
pub fn has_serde_config(project_path: &Path) -> SerdeConfigSupport {
    let cargo_toml = project_path.join("Cargo.toml");
    let mut support = SerdeConfigSupport::default();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            support.has_serde = content.contains("serde");
            support.has_serde_yaml =
                content.contains("serde_yaml") || content.contains("serde_yml");
            support.has_serde_json = content.contains("serde_json");
            support.has_toml = content.contains("toml");
            support.has_validator = content.contains("validator") || content.contains("garde");
        }
    }

    // Check for config structs with Deserialize
    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                if content.contains("#[derive") && content.contains("Deserialize") {
                    support.has_deserialize_structs = true;

                    if content.to_lowercase().contains("config") {
                        support.has_config_struct = true;
                    }
                }
            }
        }
    }

    support
}

/// Serde config support detection result.
#[derive(Debug, Clone, Default)]
pub struct SerdeConfigSupport {
    pub has_serde: bool,
    pub has_serde_yaml: bool,
    pub has_serde_json: bool,
    pub has_toml: bool,
    pub has_validator: bool,
    pub has_deserialize_structs: bool,
    pub has_config_struct: bool,
}

impl SerdeConfigSupport {
    /// Check if typed config is supported.
    pub fn has_typed_config(&self) -> bool {
        self.has_serde && self.has_config_struct
    }

    /// Check if validation is supported.
    pub fn has_validation(&self) -> bool {
        self.has_validator || self.has_serde // serde itself provides some validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FALS-AUD-001: Scripting File Audit
    // =========================================================================

    #[test]
    fn test_fals_aud_001_scripting_audit_current_project() {
        let result = audit_scripting_files(Path::new("."));
        // batuta should have no scripting files in src
        assert!(
            !result.has_violations()
                || result.matches.iter().all(|p| {
                    let s = p.to_string_lossy();
                    s.contains("scripts/") || s.contains("tests/") || s.contains("target/")
                }),
            "Found scripting violations: {:?}",
            result.matches
        );
    }

    // =========================================================================
    // FALS-AUD-002: Test Framework Audit
    // =========================================================================

    #[test]
    fn test_fals_aud_002_test_framework_audit() {
        let result = audit_test_frameworks(Path::new("."));
        // batuta should not have Jest/Pytest
        assert!(
            !result.has_violations(),
            "Found test framework violations: {:?}",
            result.matches
        );
    }

    // =========================================================================
    // FALS-AUD-003: YAML Config Audit
    // =========================================================================

    #[test]
    fn test_fals_aud_003_yaml_config_audit() {
        let _result = audit_yaml_configs(Path::new("."));
        // Just verify it runs without panic
    }

    // =========================================================================
    // FALS-AUD-004: Dependency Audit
    // =========================================================================

    #[test]
    fn test_fals_aud_004_dependency_audit_no_pyo3() {
        let result = audit_cargo_dependencies(Path::new("."), &["pyo3", "napi", "mlua"]);
        assert!(
            !result.has_violations(),
            "Found forbidden deps: {:?}",
            result.forbidden
        );
    }

    // =========================================================================
    // FALS-AUD-005: Rust Tests Detection
    // =========================================================================

    #[test]
    fn test_fals_aud_005_has_rust_tests() {
        assert!(has_rust_tests(Path::new(".")));
    }

    // =========================================================================
    // FALS-AUD-006: WASM Support Detection
    // =========================================================================

    #[test]
    fn test_fals_aud_006_wasm_support() {
        let support = has_wasm_support(Path::new("."));
        // batuta has WASM support
        assert!(support.is_supported(), "Expected WASM support");
    }

    // =========================================================================
    // FALS-AUD-007: Serde Config Detection
    // =========================================================================

    #[test]
    fn test_fals_aud_007_serde_config() {
        let support = has_serde_config(Path::new("."));
        assert!(support.has_serde, "Expected serde");
        assert!(support.has_config_struct, "Expected config struct");
    }
}
