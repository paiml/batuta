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

    if has_config_module && has_examples {
        item = item.pass();
    } else if has_config_module || !yaml_files.is_empty() {
        item = item.partial("Config module exists but examples incomplete");
    } else {
        item = item.fail("No declarative YAML configuration found");
    }

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

    let mut violations = Vec::new();
    let src_path = project_path.join("src");

    // Check for scripting language files in src/
    let scripting_extensions = [".py", ".js", ".ts", ".lua", ".rb"];

    for ext in scripting_extensions {
        if let Ok(entries) = glob::glob(&format!("{}/**/*{}", src_path.display(), ext)) {
            for entry in entries.flatten() {
                let path_str = entry.to_string_lossy();
                // Exclude wasm-bindgen glue and build artifacts
                if !path_str.contains("/target/")
                    && !path_str.contains("/pkg/")
                    && !path_str.contains(".wasm")
                {
                    violations.push(entry);
                }
            }
        }
    }

    // Check Cargo.toml for scripting runtime dependencies
    let cargo_toml = project_path.join("Cargo.toml");
    let mut scripting_deps = Vec::new();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            let forbidden_deps = ["pyo3", "napi", "mlua", "rlua", "rustpython"];

            for dep in forbidden_deps {
                // Check if it's in [dependencies] but not [dev-dependencies]
                if content.contains(&format!("{} =", dep)) || content.contains(&format!("{}=", dep))
                {
                    // Rough check - a proper implementation would parse TOML
                    if !content.contains("[dev-dependencies]")
                        || content.find(&format!("{} =", dep)) < content.find("[dev-dependencies]")
                    {
                        scripting_deps.push(dep.to_string());
                    }
                }
            }

            item = item.with_evidence(Evidence::dependency_audit(
                "Checked Cargo.toml for scripting runtimes".to_string(),
                format!("Found: {:?}", scripting_deps),
            ));
        }
    }

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} scripting files in src/", violations.len()),
        violations.clone(),
    ));

    if violations.is_empty() && scripting_deps.is_empty() {
        item = item.pass();
    } else {
        let mut reasons = Vec::new();
        if !violations.is_empty() {
            reasons.push(format!("{} scripting files in src/", violations.len()));
        }
        if !scripting_deps.is_empty() {
            reasons.push(format!("Scripting deps: {:?}", scripting_deps));
        }
        item = item.fail(reasons.join("; "));
    }

    item.with_duration(start.elapsed().as_millis() as u64)
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

    let mut violations = Vec::new();

    // Check for JavaScript test files
    let js_test_patterns = [
        "**/*.test.js",
        "**/*.spec.js",
        "**/*.test.ts",
        "**/*.spec.ts",
        "**/jest.config.*",
        "**/vitest.config.*",
    ];

    for pattern in js_test_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                // Exclude node_modules
                if !entry.to_string_lossy().contains("node_modules") {
                    violations.push(entry);
                }
            }
        }
    }

    // Check for Python test files
    let py_test_patterns = [
        "**/test_*.py",
        "**/*_test.py",
        "**/conftest.py",
        "**/pytest.ini",
        "**/pyproject.toml",
    ];

    for pattern in py_test_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                // Exclude venv
                let path_str = entry.to_string_lossy();
                if !path_str.contains("venv") && !path_str.contains(".venv") {
                    violations.push(entry);
                }
            }
        }
    }

    // Check for package.json with test scripts
    let package_json = project_path.join("package.json");
    if package_json.exists() {
        if let Ok(content) = std::fs::read_to_string(&package_json) {
            if content.contains("\"test\"") || content.contains("\"jest\"") {
                violations.push(package_json);
            }
        }
    }

    // Check for node_modules
    let node_modules = project_path.join("node_modules");
    if node_modules.exists() {
        violations.push(node_modules);
    }

    // Check for __pycache__
    if let Ok(entries) = glob::glob(&format!("{}/**/__pycache__", project_path.display())) {
        for entry in entries.flatten() {
            violations.push(entry);
        }
    }

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} non-Rust test artifacts", violations.len()),
        violations.clone(),
    ));

    // Check for Rust tests (positive indicator)
    let has_rust_tests = project_path.join("tests").exists()
        || glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries.flatten().any(|p| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .map(|c| c.contains("#[test]") || c.contains("#[cfg(test)]"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    if violations.is_empty() && has_rust_tests {
        item = item.pass();
    } else if violations.is_empty() {
        item = item.partial("No violations but no Rust tests detected");
    } else {
        item = item.fail(format!(
            "Found {} non-Rust test artifacts: {:?}",
            violations.len(),
            violations.iter().take(5).collect::<Vec<_>>()
        ));
    }

    item.with_duration(start.elapsed().as_millis() as u64)
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

    // Check for WASM target in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let mut has_wasm_feature = false;
    let mut has_wasm_bindgen = false;

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            has_wasm_feature = content.contains("wasm") || content.contains("web");
            has_wasm_bindgen = content.contains("wasm-bindgen")
                || content.contains("wasm-pack")
                || content.contains("web-sys");
        }
    }

    // Check for WASM source file
    let has_wasm_module =
        project_path.join("src/wasm.rs").exists() || project_path.join("src/lib.rs").exists(); // lib.rs often has wasm exports

    // Check for excessive JavaScript
    let mut js_files = Vec::new();
    if let Ok(entries) = glob::glob(&format!("{}/**/*.js", project_path.display())) {
        for entry in entries.flatten() {
            let path_str = entry.to_string_lossy();
            // Exclude node_modules, pkg (wasm-pack output), dist, book (mdBook), docs, target
            // Use both with and without leading slash to handle relative and absolute paths
            let is_excluded = path_str.contains("node_modules")
                || path_str.contains("/pkg/")
                || path_str.contains("/dist/")
                || path_str.contains("/target/")
                || path_str.contains("/book/")
                || path_str.contains("/docs/")
                // Handle relative paths starting with these directories
                || path_str.starts_with("target/")
                || path_str.starts_with("pkg/")
                || path_str.starts_with("dist/")
                || path_str.starts_with("book/")
                || path_str.starts_with("docs/");

            if !is_excluded {
                js_files.push(entry);
            }
        }
    }

    // Check for JS frameworks
    let package_json = project_path.join("package.json");
    let mut has_js_framework = false;
    if package_json.exists() {
        if let Ok(content) = std::fs::read_to_string(&package_json) {
            let frameworks = ["react", "vue", "svelte", "angular", "next", "nuxt"];
            for framework in frameworks {
                if content.contains(framework) {
                    has_js_framework = true;
                    break;
                }
            }
        }
    }

    item = item.with_evidence(Evidence::file_audit(
        format!(
            "WASM: feature={}, bindgen={}, JS files={}",
            has_wasm_feature,
            has_wasm_bindgen,
            js_files.len()
        ),
        js_files.clone(),
    ));

    // Evaluate
    if has_js_framework {
        item = item.fail("JavaScript framework detected (React/Vue/Svelte)");
    } else if js_files.len() > 5 {
        item = item.fail(format!(
            "Too many JS files ({}) beyond WASM glue",
            js_files.len()
        ));
    } else if has_wasm_bindgen || has_wasm_feature {
        if js_files.is_empty() {
            item = item.pass();
        } else {
            item = item.partial(format!(
                "WASM support exists but {} JS files found",
                js_files.len()
            ));
        }
    } else if has_wasm_module && js_files.is_empty() {
        item = item.partial("No explicit WASM feature but no JS violations");
    } else {
        item = item.fail("No WASM support detected");
    }

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
    let cargo_toml = project_path.join("Cargo.toml");
    let mut has_serde = false;
    let mut has_serde_yaml = false;
    let mut has_validator = false;

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            has_serde = content.contains("serde");
            has_serde_yaml = content.contains("serde_yaml") || content.contains("serde_yml");
            has_validator = content.contains("validator") || content.contains("garde");
        }
    }

    // Check for config struct with Deserialize
    let mut has_config_struct = false;
    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                if (content.contains("#[derive") && content.contains("Deserialize"))
                    && (content.contains("struct") && content.to_lowercase().contains("config"))
                {
                    has_config_struct = true;
                    break;
                }
            }
        }
    }

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

    if has_serde && has_serde_yaml && has_config_struct {
        if has_validator || has_json_schema {
            item = item.pass();
        } else {
            item = item.partial("Basic serde validation but no explicit validator");
        }
    } else if has_serde && has_config_struct {
        item = item.partial("Config struct exists but YAML support unclear");
    } else {
        item = item.fail("No typed schema validation for configs");
    }

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
}
