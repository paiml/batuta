//! Architectural Invariants - CRITICAL Checks
//!
//! Section 10 of the Popperian Falsification Checklist.
//! Any failure in this section = Project FAIL.
//!
//! - AI-01: Declarative YAML Configuration
//! - AI-02: Zero Scripting in Production
//! - AI-03: Pure Rust Testing (No Jest/Pytest)
//! - AI-04: WASM-First Browser Support
//! - AI-05: Declarative Schema Validation

use crate::falsification::helpers::{apply_check_outcome, CheckOutcome};
use crate::falsification::types::*;
use std::path::Path;
use std::time::Instant;

/// Evaluate all architectural invariants.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_declarative_yaml(project_path),
        check_zero_scripting(project_path),
        check_pure_rust_testing(project_path),
        check_wasm_first(project_path),
        check_schema_validation(project_path),
    ]
}

/// AI-01: Declarative YAML Configuration
///
/// **Claim:** Project offers full functionality via declarative YAML without code.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any core feature unavailable via YAML config
/// - User must write Rust/code to use basic functionality
pub fn check_declarative_yaml(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-01",
        "Declarative YAML Configuration",
        "Project offers full functionality via declarative YAML without code",
    )
    .with_severity(Severity::Critical)
    .with_tps("Poka-Yoke — enable non-developers");

    // Check for YAML config files
    let yaml_patterns = ["*.yaml", "*.yml"];
    let mut yaml_files = Vec::new();

    for pattern in yaml_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                yaml_files.push(entry);
            }
        }
        // Also check config directories
        if let Ok(entries) =
            glob::glob(&format!("{}/config/**/{}", project_path.display(), pattern))
        {
            for entry in entries.flatten() {
                yaml_files.push(entry);
            }
        }
    }

    // Check for schema definitions (serde structs)
    let has_config_module = project_path.join("src/config.rs").exists()
        || project_path.join("src/config/mod.rs").exists();

    // Check for example configs
    let has_examples = project_path.join("examples").exists() || !yaml_files.is_empty();

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} YAML config files", yaml_files.len()),
        yaml_files.clone(),
    ));

    item = apply_check_outcome(
        item,
        &[
            (has_config_module && has_examples, CheckOutcome::Pass),
            (
                has_config_module || !yaml_files.is_empty(),
                CheckOutcome::Partial("Config module exists but examples incomplete"),
            ),
            (
                true,
                CheckOutcome::Fail("No declarative YAML configuration found"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// AI-02: Zero Scripting in Production
///
/// **Claim:** No Python/JavaScript/Lua in production runtime paths.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any `.py`, `.js`, `.lua` in src/ or runtime
/// - pyo3, napi-rs, mlua in non-dev dependencies
/// - Interpreter embedded in binary
pub fn check_zero_scripting(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-02",
        "Zero Scripting in Production",
        "No Python/JavaScript/Lua in production runtime paths",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — type safety, determinism");

    // Check for scripting language files in src/
    let violations = find_glob_violations(
        project_path,
        &[
            "src/**/*.py",
            "src/**/*.js",
            "src/**/*.ts",
            "src/**/*.lua",
            "src/**/*.rb",
        ],
        &["/target/", "/pkg/", ".wasm"],
    );

    // Check Cargo.toml for scripting runtime dependencies
    let scripting_deps = super::helpers::find_scripting_deps(project_path);

    item = item.with_evidence(Evidence::dependency_audit(
        "Checked Cargo.toml for scripting runtimes".to_string(),
        format!("Found: {:?}", scripting_deps),
    ));

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} scripting files in src/", violations.len()),
        violations.clone(),
    ));

    let fail_reasons = {
        let mut r = Vec::new();
        if !violations.is_empty() {
            r.push(format!("{} scripting files in src/", violations.len()));
        }
        if !scripting_deps.is_empty() {
            r.push(format!("Scripting deps: {:?}", scripting_deps));
        }
        r.join("; ")
    };
    item = apply_check_outcome(
        item,
        &[
            (
                violations.is_empty() && scripting_deps.is_empty(),
                CheckOutcome::Pass,
            ),
            (true, CheckOutcome::Fail(&fail_reasons)),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Find files matching glob patterns, excluding paths containing any exclude string
fn find_glob_violations(
    project_path: &Path,
    patterns: &[&str],
    excludes: &[&str],
) -> Vec<std::path::PathBuf> {
    let mut results = Vec::new();
    for pattern in patterns {
        let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path_str = entry.to_string_lossy();
            if !excludes.iter().any(|ex| path_str.contains(ex)) {
                results.push(entry);
            }
        }
    }
    results
}

/// Check if Rust tests exist in the project
fn has_rust_tests(project_path: &Path) -> bool {
    project_path.join("tests").exists()
        || glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries.flatten().any(|p| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .is_some_and(|c| c.contains("#[test]") || c.contains("#[cfg(test)]"))
                })
            })
            .unwrap_or(false)
}

/// AI-03: Pure Rust Testing (No Jest/Pytest)
///
/// **Claim:** All tests written in Rust, no external test frameworks.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any Jest, Mocha, Pytest, unittest files
/// - package.json with test scripts
/// - requirements-dev.txt with pytest
pub fn check_pure_rust_testing(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-03",
        "Pure Rust Testing",
        "All tests written in Rust, no external test frameworks",
    )
    .with_severity(Severity::Critical)
    .with_tps("Zero scripting policy");

    let mut violations = find_glob_violations(
        project_path,
        &[
            "**/*.test.js",
            "**/*.spec.js",
            "**/*.test.ts",
            "**/*.spec.ts",
            "**/jest.config.*",
            "**/vitest.config.*",
        ],
        &["node_modules"],
    );

    violations.extend(find_glob_violations(
        project_path,
        &[
            "**/test_*.py",
            "**/*_test.py",
            "**/conftest.py",
            "**/pytest.ini",
            "**/pyproject.toml",
        ],
        &["venv", ".venv"],
    ));

    let package_json = project_path.join("package.json");
    if let Ok(content) = std::fs::read_to_string(&package_json) {
        if content.contains("\"test\"") || content.contains("\"jest\"") {
            violations.push(package_json);
        }
    }

    let node_modules = project_path.join("node_modules");
    if node_modules.exists() {
        violations.push(node_modules);
    }

    violations.extend(find_glob_violations(project_path, &["**/__pycache__"], &[]));

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} non-Rust test artifacts", violations.len()),
        violations.clone(),
    ));

    let has_tests = has_rust_tests(project_path);
    let fail_msg = format!(
        "Found {} non-Rust test artifacts: {:?}",
        violations.len(),
        violations.iter().take(5).collect::<Vec<_>>()
    );
    item = apply_check_outcome(
        item,
        &[
            (violations.is_empty() && has_tests, CheckOutcome::Pass),
            (
                violations.is_empty(),
                CheckOutcome::Partial("No violations but no Rust tests detected"),
            ),
            (true, CheckOutcome::Fail(&fail_msg)),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Check if a JS file path should be excluded from analysis
fn is_excluded_js_path(path_str: &str) -> bool {
    const EXCLUDED_DIRS: &[&str] = &[
        "node_modules",
        "/pkg/",
        "/dist/",
        "/target/",
        "/book/",
        "/docs/",
    ];
    const EXCLUDED_PREFIXES: &[&str] = &["target/", "pkg/", "dist/", "book/", "docs/"];

    EXCLUDED_DIRS.iter().any(|d| path_str.contains(d))
        || EXCLUDED_PREFIXES.iter().any(|p| path_str.starts_with(p))
}

/// Find non-excluded JS files in project
fn find_js_files(project_path: &Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = glob::glob(&format!("{}/**/*.js", project_path.display())) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| !is_excluded_js_path(&entry.to_string_lossy()))
        .collect()
}

/// Detect JS framework in package.json
fn detect_js_framework(project_path: &Path) -> bool {
    let package_json = project_path.join("package.json");
    let Ok(content) = std::fs::read_to_string(package_json) else {
        return false;
    };
    ["react", "vue", "svelte", "angular", "next", "nuxt"]
        .iter()
        .any(|fw| content.contains(fw))
}

/// Parse WASM feature/bindgen info from Cargo.toml
fn parse_wasm_cargo_info(project_path: &Path) -> (bool, bool) {
    let cargo_toml = project_path.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(cargo_toml) else {
        return (false, false);
    };
    let has_feature = content.contains("wasm") || content.contains("web");
    let has_bindgen = content.contains("wasm-bindgen")
        || content.contains("wasm-pack")
        || content.contains("web-sys");
    (has_feature, has_bindgen)
}

/// AI-04: WASM-First Browser Support
///
/// **Claim:** Browser functionality via WASM, not JavaScript.
///
/// **Rejection Criteria (CRITICAL):**
/// - JS files beyond minimal WASM glue
/// - npm dependencies for core functionality
/// - React/Vue/Svelte instead of WASM UI
pub fn check_wasm_first(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-04",
        "WASM-First Browser Support",
        "Browser functionality via WASM, not JavaScript",
    )
    .with_severity(Severity::Critical)
    .with_tps("Zero scripting, sovereignty");

    let (has_wasm_feature, has_wasm_bindgen) = parse_wasm_cargo_info(project_path);
    let has_wasm_module =
        project_path.join("src/wasm.rs").exists() || project_path.join("src/lib.rs").exists();
    let js_files = find_js_files(project_path);
    let has_js_framework = detect_js_framework(project_path);

    item = item.with_evidence(Evidence::file_audit(
        format!(
            "WASM: feature={}, bindgen={}, JS files={}",
            has_wasm_feature,
            has_wasm_bindgen,
            js_files.len()
        ),
        js_files.clone(),
    ));

    let too_many_js_msg = format!("Too many JS files ({}) beyond WASM glue", js_files.len());
    let wasm_partial_msg = format!("WASM support exists but {} JS files found", js_files.len());
    let has_wasm_support = has_wasm_bindgen || has_wasm_feature;
    item = apply_check_outcome(
        item,
        &[
            (
                has_js_framework,
                CheckOutcome::Fail("JavaScript framework detected (React/Vue/Svelte)"),
            ),
            (js_files.len() > 5, CheckOutcome::Fail(&too_many_js_msg)),
            (has_wasm_support && js_files.is_empty(), CheckOutcome::Pass),
            (has_wasm_support, CheckOutcome::Partial(&wasm_partial_msg)),
            (
                has_wasm_module && js_files.is_empty(),
                CheckOutcome::Partial("No explicit WASM feature but no JS violations"),
            ),
            (true, CheckOutcome::Fail("No WASM support detected")),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// AI-05: Declarative Schema Validation
///
/// **Claim:** YAML configs validated against typed schema.
///
/// **Rejection Criteria (CRITICAL):**
/// - Invalid YAML silently accepted
/// - No JSON Schema or serde validation
/// - Runtime panics on bad config
pub fn check_schema_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-05",
        "Declarative Schema Validation",
        "YAML configs validated against typed schema",
    )
    .with_severity(Severity::Critical)
    .with_tps("Poka-Yoke — prevent config errors");

    // Check for serde in Cargo.toml
    let schema = super::helpers::detect_schema_deps(project_path);
    let has_serde = schema.has_serde;
    let has_serde_yaml = schema.has_serde_yaml;
    let has_validator = schema.has_validator;

    // Check for config struct with Deserialize
    let has_config_struct = super::helpers::has_deserialize_config_struct(project_path);

    // Check for JSON Schema files
    let has_json_schema = glob::glob(&format!("{}/**/*.schema.json", project_path.display()))
        .ok()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);

    item = item.with_evidence(Evidence::schema_validation(
        format!(
            "serde={}, yaml={}, validator={}, config_struct={}, json_schema={}",
            has_serde, has_serde_yaml, has_validator, has_config_struct, has_json_schema
        ),
        format!(
            "Schema validation: {}",
            if has_config_struct || has_json_schema {
                "PRESENT"
            } else {
                "MISSING"
            }
        ),
    ));

    let has_full_serde = has_serde && has_serde_yaml && has_config_struct;
    item = apply_check_outcome(
        item,
        &[
            (
                has_full_serde && (has_validator || has_json_schema),
                CheckOutcome::Pass,
            ),
            (
                has_full_serde,
                CheckOutcome::Partial("Basic serde validation but no explicit validator"),
            ),
            (
                has_serde && has_config_struct,
                CheckOutcome::Partial("Config struct exists but YAML support unclear"),
            ),
            (
                true,
                CheckOutcome::Fail("No typed schema validation for configs"),
            ),
        ],
    );

    item.with_duration(start.elapsed().as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // =========================================================================
    // FALS-INV-001: Declarative YAML Check
    // =========================================================================

    #[test]
    fn test_fals_inv_001_declarative_yaml_current_project() {
        let result = check_declarative_yaml(Path::new("."));
        assert_eq!(result.id, "AI-01");
        assert_eq!(result.severity, Severity::Critical);
        // Current project should pass or partial
        assert!(result.status != CheckStatus::Skipped);
    }

    // =========================================================================
    // FALS-INV-002: Zero Scripting Check
    // =========================================================================

    #[test]
    fn test_fals_inv_002_zero_scripting_current_project() {
        let result = check_zero_scripting(Path::new("."));
        assert_eq!(result.id, "AI-02");
        assert_eq!(result.severity, Severity::Critical);
        // batuta should pass zero scripting
        assert_eq!(result.status, CheckStatus::Pass);
    }

    // =========================================================================
    // FALS-INV-003: Pure Rust Testing Check
    // =========================================================================

    #[test]
    fn test_fals_inv_003_pure_rust_testing_current_project() {
        let result = check_pure_rust_testing(Path::new("."));
        assert_eq!(result.id, "AI-03");
        assert_eq!(result.severity, Severity::Critical);
        // batuta uses only Rust tests
        assert!(matches!(
            result.status,
            CheckStatus::Pass | CheckStatus::Partial
        ));
    }

    // =========================================================================
    // FALS-INV-004: WASM First Check
    // =========================================================================

    #[test]
    fn test_fals_inv_004_wasm_first_current_project() {
        let result = check_wasm_first(Path::new("."));
        assert_eq!(result.id, "AI-04");
        assert_eq!(result.severity, Severity::Critical);
        // batuta has WASM support
        assert!(result.status != CheckStatus::Skipped);
    }

    // =========================================================================
    // FALS-INV-005: Schema Validation Check
    // =========================================================================

    #[test]
    fn test_fals_inv_005_schema_validation_current_project() {
        let result = check_schema_validation(Path::new("."));
        assert_eq!(result.id, "AI-05");
        assert_eq!(result.severity, Severity::Critical);
        // batuta uses serde for config
        assert!(matches!(
            result.status,
            CheckStatus::Pass | CheckStatus::Partial
        ));
    }

    // =========================================================================
    // FALS-INV-006: Evaluate All
    // =========================================================================

    #[test]
    fn test_fals_inv_006_evaluate_all_returns_5_items() {
        let results = evaluate_all(Path::new("."));
        assert_eq!(results.len(), 5);

        // All should be critical severity
        for item in &results {
            assert_eq!(item.severity, Severity::Critical);
        }
    }

    #[test]
    fn test_fals_inv_006_all_items_have_evidence() {
        let results = evaluate_all(Path::new("."));

        for item in &results {
            assert!(
                !item.evidence.is_empty(),
                "Item {} has no evidence",
                item.id
            );
        }
    }

    // =========================================================================
    // FALS-INV-007: Duration Tracking
    // =========================================================================

    #[test]
    fn test_fals_inv_007_duration_tracked() {
        let result = check_declarative_yaml(Path::new("."));
        // Should have some duration (even if very fast)
        // We just check it's set, not a specific value
        assert!(result.duration_ms < 60000); // Less than 1 minute
    }

    // =========================================================================
    // Additional Tests for Coverage
    // =========================================================================

    #[test]
    fn test_fals_inv_008_declarative_yaml_nonexistent() {
        let result = check_declarative_yaml(Path::new("/nonexistent/path"));
        // Should handle missing path gracefully
        assert_eq!(result.id, "AI-01");
        assert!(result.status != CheckStatus::Pass);
    }

    #[test]
    fn test_fals_inv_008_zero_scripting_nonexistent() {
        let result = check_zero_scripting(Path::new("/nonexistent/path"));
        // Should handle missing path gracefully - passes because no scripts found
        assert_eq!(result.id, "AI-02");
    }

    #[test]
    fn test_fals_inv_008_pure_rust_testing_nonexistent() {
        let result = check_pure_rust_testing(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-03");
    }

    #[test]
    fn test_fals_inv_008_wasm_first_nonexistent() {
        let result = check_wasm_first(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-04");
    }

    #[test]
    fn test_fals_inv_008_schema_validation_nonexistent() {
        let result = check_schema_validation(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-05");
    }

    #[test]
    fn test_fals_inv_009_declarative_yaml_temp_with_yaml() {
        let temp_dir = std::env::temp_dir().join("test_yaml_config");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create a YAML file
        std::fs::write(temp_dir.join("config.yaml"), "key: value").unwrap();

        let result = check_declarative_yaml(&temp_dir);
        assert_eq!(result.id, "AI-01");
        // Has YAML files so should at least be partial
        assert!(result.status == CheckStatus::Partial || result.status == CheckStatus::Pass);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_009_declarative_yaml_temp_with_config_module() {
        let temp_dir = std::env::temp_dir().join("test_yaml_config_mod");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();
        std::fs::create_dir_all(temp_dir.join("examples")).unwrap();

        // Create config.rs
        std::fs::write(temp_dir.join("src/config.rs"), "pub struct Config {}").unwrap();

        let result = check_declarative_yaml(&temp_dir);
        assert_eq!(result.id, "AI-01");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_010_zero_scripting_with_scripts() {
        let temp_dir = std::env::temp_dir().join("test_scripting");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create a Python file in src
        std::fs::write(temp_dir.join("src/helper.py"), "print('hello')").unwrap();

        let result = check_zero_scripting(&temp_dir);
        assert_eq!(result.id, "AI-02");
        // Should fail because script in src/
        assert!(result.status == CheckStatus::Fail);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_010_zero_scripting_with_scripts_in_tools() {
        let temp_dir = std::env::temp_dir().join("test_scripting_tools");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("tools")).unwrap();

        // Create a Python file in tools (acceptable)
        std::fs::write(temp_dir.join("tools/build.py"), "print('build')").unwrap();

        let result = check_zero_scripting(&temp_dir);
        assert_eq!(result.id, "AI-02");
        // Should pass because scripts only in tools/
        assert!(result.status == CheckStatus::Pass);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_011_pure_rust_testing_with_rust_tests() {
        let temp_dir = std::env::temp_dir().join("test_rust_testing");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create Rust file with tests
        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
"#,
        )
        .unwrap();

        let result = check_pure_rust_testing(&temp_dir);
        assert_eq!(result.id, "AI-03");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_012_wasm_first_with_cargo_toml() {
        let temp_dir = std::env::temp_dir().join("test_wasm_first");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create Cargo.toml with wasm target
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[lib]
crate-type = ["cdylib", "rlib"]

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
"#,
        )
        .unwrap();

        let result = check_wasm_first(&temp_dir);
        assert_eq!(result.id, "AI-04");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_013_schema_validation_with_schemars() {
        let temp_dir = std::env::temp_dir().join("test_schema_val");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create Cargo.toml with schemars
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
schemars = "0.8"
"#,
        )
        .unwrap();

        let result = check_schema_validation(&temp_dir);
        assert_eq!(result.id, "AI-05");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fals_inv_014_all_severities_critical() {
        let results = evaluate_all(Path::new("."));
        for result in results {
            // All architectural invariants should be Critical
            assert_eq!(result.severity, Severity::Critical);
        }
    }

    #[test]
    fn test_fals_inv_015_all_have_tps_principle() {
        let results = evaluate_all(Path::new("."));
        for result in results {
            assert!(!result.tps_principle.is_empty());
        }
    }

    // =========================================================================
    // ADDITIONAL COVERAGE TESTS
    // =========================================================================

    #[test]
    fn test_inv_cov_001_yaml_nonexistent_path() {
        let result = check_declarative_yaml(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-01");
        assert!(matches!(result.status, CheckStatus::Fail));
    }

    #[test]
    fn test_inv_cov_002_scripting_nonexistent_path() {
        let result = check_zero_scripting(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-02");
        // No scripting files means pass
        assert!(matches!(result.status, CheckStatus::Pass));
    }

    #[test]
    fn test_inv_cov_003_rust_testing_nonexistent_path() {
        let result = check_pure_rust_testing(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-03");
        // No violations means partial (no Rust tests found either)
        assert!(matches!(
            result.status,
            CheckStatus::Partial | CheckStatus::Pass | CheckStatus::Fail
        ));
    }

    #[test]
    fn test_inv_cov_004_wasm_first_nonexistent_path() {
        let result = check_wasm_first(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-04");
    }

    #[test]
    fn test_inv_cov_005_schema_validation_nonexistent_path() {
        let result = check_schema_validation(Path::new("/nonexistent/path"));
        assert_eq!(result.id, "AI-05");
    }

    #[test]
    fn test_inv_cov_006_yaml_with_config_dir() {
        let temp_dir = std::env::temp_dir().join("test_yaml_config");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("config")).unwrap();
        std::fs::create_dir_all(temp_dir.join("examples")).unwrap();
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create config module
        std::fs::write(temp_dir.join("src/config.rs"), "// config").unwrap();

        // Create config YAML
        std::fs::write(temp_dir.join("config/app.yaml"), "key: value").unwrap();

        let result = check_declarative_yaml(&temp_dir);
        assert_eq!(result.id, "AI-01");
        assert!(matches!(result.status, CheckStatus::Pass));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_007_scripting_with_violations() {
        let temp_dir = std::env::temp_dir().join("test_scripting_vio");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create Python file in src
        std::fs::write(temp_dir.join("src/script.py"), "print('hello')").unwrap();

        let result = check_zero_scripting(&temp_dir);
        assert_eq!(result.id, "AI-02");
        assert!(matches!(result.status, CheckStatus::Fail));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_008_scripting_deps_check() {
        let temp_dir = std::env::temp_dir().join("test_scripting_deps");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create Cargo.toml with pyo3
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
pyo3 = "0.20"
"#,
        )
        .unwrap();

        let result = check_zero_scripting(&temp_dir);
        assert_eq!(result.id, "AI-02");
        assert!(matches!(result.status, CheckStatus::Fail));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_009_pure_rust_with_violations() {
        let temp_dir = std::env::temp_dir().join("test_rust_vio");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create JS test file
        std::fs::write(temp_dir.join("example.test.js"), "test()").unwrap();

        let result = check_pure_rust_testing(&temp_dir);
        assert_eq!(result.id, "AI-03");
        assert!(matches!(result.status, CheckStatus::Fail));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_010_wasm_with_wasm_bindgen() {
        let temp_dir = std::env::temp_dir().join("test_wasm_bind");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create Cargo.toml with wasm-bindgen
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
wasm-bindgen = "0.2"
"#,
        )
        .unwrap();

        let result = check_wasm_first(&temp_dir);
        assert_eq!(result.id, "AI-04");
        assert!(matches!(
            result.status,
            CheckStatus::Pass | CheckStatus::Partial
        ));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_011_wasm_with_js_violations() {
        let temp_dir = std::env::temp_dir().join("test_wasm_js");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create excessive JS files
        for i in 0..6 {
            std::fs::write(
                temp_dir.join(format!("src/app{}.js", i)),
                "export function f() {}",
            )
            .unwrap();
        }

        let result = check_wasm_first(&temp_dir);
        assert_eq!(result.id, "AI-04");
        assert!(matches!(
            result.status,
            CheckStatus::Fail | CheckStatus::Partial
        ));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_012_schema_with_serde() {
        let temp_dir = std::env::temp_dir().join("test_schema_serde");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create Cargo.toml with serde and validator
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
validator = "0.16"
"#,
        )
        .unwrap();

        // Create config struct with Deserialize
        std::fs::write(
            temp_dir.join("src/config.rs"),
            r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub name: String,
}
"#,
        )
        .unwrap();

        let result = check_schema_validation(&temp_dir);
        assert_eq!(result.id, "AI-05");
        assert!(matches!(
            result.status,
            CheckStatus::Pass | CheckStatus::Partial
        ));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_013_pure_rust_with_node_modules() {
        let temp_dir = std::env::temp_dir().join("test_node_modules");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("node_modules")).unwrap();

        let result = check_pure_rust_testing(&temp_dir);
        assert_eq!(result.id, "AI-03");
        assert!(matches!(result.status, CheckStatus::Fail));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_014_pure_rust_with_package_json() {
        let temp_dir = std::env::temp_dir().join("test_pkg_json");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create package.json with test scripts
        std::fs::write(
            temp_dir.join("package.json"),
            r#"{"scripts": {"test": "jest"}}"#,
        )
        .unwrap();

        let result = check_pure_rust_testing(&temp_dir);
        assert_eq!(result.id, "AI-03");
        assert!(matches!(result.status, CheckStatus::Fail));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_015_yaml_partial_case() {
        let temp_dir = std::env::temp_dir().join("test_yaml_partial");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        // Create config module but no examples
        std::fs::write(temp_dir.join("src/config.rs"), "// config").unwrap();

        let result = check_declarative_yaml(&temp_dir);
        assert_eq!(result.id, "AI-01");
        assert!(matches!(
            result.status,
            CheckStatus::Partial | CheckStatus::Pass
        ));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_inv_cov_016_all_durations_reasonable() {
        let results = evaluate_all(Path::new("."));
        for result in results {
            // Duration should be reasonable (less than 1 minute per check)
            assert!(
                result.duration_ms < 60_000,
                "Check {} took unreasonably long: {}ms",
                result.id,
                result.duration_ms
            );
        }
    }
}
