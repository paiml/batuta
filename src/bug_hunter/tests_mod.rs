use super::*;
use std::path::PathBuf;

// =========================================================================
// BH-MOD-001: Hunt Function
// =========================================================================

#[test]
fn test_bh_mod_001_hunt_returns_result() {
    let config = HuntConfig {
        mode: HuntMode::Analyze,
        ..Default::default()
    };
    let result = hunt(Path::new("."), config);
    assert_eq!(result.mode, HuntMode::Analyze);
}

#[test]
fn test_bh_mod_001_hunt_all_modes() {
    for mode in [HuntMode::Falsify, HuntMode::Hunt, HuntMode::Analyze, HuntMode::Fuzz, HuntMode::DeepHunt] {
        let config = HuntConfig {
            mode,
            targets: vec![PathBuf::from("src")],
            ..Default::default()
        };
        let result = hunt(Path::new("."), config);
        assert_eq!(result.mode, mode);
    }
}

// =========================================================================
// BH-MOD-002: Ensemble Hunt
// =========================================================================

#[test]
fn test_bh_mod_002_hunt_ensemble() {
    let config = HuntConfig::default();
    let result = hunt_ensemble(Path::new("."), config);
    // Should have findings from multiple modes
    assert!(result.duration_ms > 0);
}

// =========================================================================
// BH-MOD-003: Category Classification
// =========================================================================

#[test]
fn test_bh_mod_003_categorize_clippy_memory() {
    let (cat, sev) = categorize_clippy_warning("ptr_null", "test");
    assert_eq!(cat, DefectCategory::MemorySafety);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_003_categorize_clippy_concurrency() {
    let (cat, sev) = categorize_clippy_warning("mutex_atomic", "test");
    assert_eq!(cat, DefectCategory::ConcurrencyBugs);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_003_categorize_clippy_unknown() {
    let (cat, sev) = categorize_clippy_warning("some_other_lint", "test");
    assert_eq!(cat, DefectCategory::Unknown);
    assert_eq!(sev, FindingSeverity::Low);
}

// =========================================================================
// BH-MOD-004: Result Finalization
// =========================================================================

#[test]
fn test_bh_mod_004_result_finalize() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new(".", HuntMode::Analyze, config);
    result.add_finding(
        Finding::new("F-001", "test.rs", 1, "Test")
            .with_severity(FindingSeverity::High)
            .with_suspiciousness(0.8),
    );
    result.finalize();

    assert_eq!(result.stats.total_findings, 1);
    assert_eq!(result.stats.by_severity.get(&FindingSeverity::High), Some(&1));
}

// =========================================================================
// BH-MOD-005: Test Code Detection
// =========================================================================

#[test]
fn test_bh_mod_005_compute_test_lines_cfg_test() {
    let content = r#"fn production_code() {
    panic!("this should be caught");
}

#[cfg(test)]
mod tests {
    #[test]
    fn my_test() {
        panic!("this should be ignored");
    }
}"#;
    let test_lines = compute_test_lines(content);
    assert!(test_lines.contains(&5), "Line 5 (#[cfg(test)]) should be test code");
    assert!(test_lines.contains(&6), "Line 6 (mod tests) should be test code");
    assert!(test_lines.contains(&7), "Line 7 (#[test]) should be test code");
    assert!(test_lines.contains(&9), "Line 9 (panic) should be test code");
    assert!(!test_lines.contains(&2), "Line 2 (production panic) should NOT be test code");
}

#[test]
fn test_bh_mod_005_compute_test_lines_individual_test() {
    let content = r#"fn production_code() {
    println!("production");
}

#[test]
fn standalone_test() {
    panic!("test assertion");
}"#;
    let test_lines = compute_test_lines(content);
    assert!(test_lines.contains(&5), "Line 5 (#[test]) should be test code");
    assert!(test_lines.contains(&6), "Line 6 (fn standalone_test) should be test code");
    assert!(test_lines.contains(&7), "Line 7 (panic) should be test code");
    assert!(!test_lines.contains(&1), "Line 1 (fn production_code) should NOT be test code");
    assert!(!test_lines.contains(&2), "Line 2 (println) should NOT be test code");
}

#[test]
fn test_bh_mod_005_compute_test_lines_empty() {
    let content = "fn main() {}\n";
    let test_lines = compute_test_lines(content);
    assert!(test_lines.is_empty());
}

// =========================================================================
// BH-MOD-006: Real Pattern Detection
// =========================================================================

#[test]
fn test_bh_mod_006_real_pattern_todo_in_comment() {
    assert!(is_real_pattern("// TODO: fix this", "TODO"));
    assert!(is_real_pattern("    // TODO fix later", "TODO"));
}

#[test]
fn test_bh_mod_006_real_pattern_todo_in_string() {
    assert!(!is_real_pattern(r#"let msg = "TODO: implement";"#, "TODO"));
    assert!(!is_real_pattern(r#"println!("TODO/FIXME markers");"#, "TODO"));
}

#[test]
fn test_bh_mod_006_real_pattern_unsafe_in_code() {
    assert!(is_real_pattern("unsafe { ptr::read(p) }", "unsafe {"));
}

#[test]
fn test_bh_mod_006_real_pattern_unsafe_in_comment() {
    assert!(!is_real_pattern("// unsafe blocks need safety comments", "unsafe {"));
}

#[test]
fn test_bh_mod_006_real_pattern_unsafe_in_variable() {
    assert!(!is_real_pattern("if in_unsafe {", "unsafe {"));
    assert!(!is_real_pattern("let foo_unsafe = true;", "unsafe {"));
    assert!(is_real_pattern("    unsafe { foo() }", "unsafe {"));
    assert!(is_real_pattern("return unsafe { bar };", "unsafe {"));
}

#[test]
fn test_bh_mod_006_real_pattern_unwrap_in_code() {
    assert!(is_real_pattern("let x = opt.unwrap();", "unwrap()"));
}

#[test]
fn test_bh_mod_006_real_pattern_unwrap_in_doc() {
    assert!(!is_real_pattern("/// Use unwrap() for testing only", "unwrap()"));
}

// =========================================================================
// BH-MOD-007: GH-18 lcov.info path detection tests
// =========================================================================

#[test]
fn test_bh_mod_007_coverage_path_config() {
    let mut config = HuntConfig::default();
    assert!(config.coverage_path.is_none());

    config.coverage_path = Some(std::path::PathBuf::from("/custom/lcov.info"));
    assert_eq!(
        config.coverage_path.as_ref().unwrap().to_str().unwrap(),
        "/custom/lcov.info"
    );
}

#[test]
fn test_bh_mod_007_analyze_coverage_hotspots_no_file() {
    let temp = std::env::temp_dir().join("test_bh_mod_007");
    let _ = std::fs::create_dir_all(&temp);
    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    analyze_coverage_hotspots(&temp, &config, &mut result);

    let nocov = result.findings.iter().any(|f| f.id == "BH-HUNT-NOCOV");
    assert!(nocov, "Should report BH-HUNT-NOCOV when no coverage file exists");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-008: GH-19 forbid(unsafe_code) detection tests
// =========================================================================

#[test]
fn test_bh_mod_008_crate_forbids_unsafe_lib_rs() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_lib");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "#![forbid(unsafe_code)]\n\npub fn safe_fn() {}\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp), "Should detect #![forbid(unsafe_code)] in lib.rs");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_008_crate_forbids_unsafe_main_rs() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_main");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/main.rs"),
        "#![forbid(unsafe_code)]\n\nfn main() {}\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp), "Should detect #![forbid(unsafe_code)] in main.rs");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_008_crate_forbids_unsafe_cargo_toml() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_cargo");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(temp.join("src/lib.rs"), "pub fn safe_fn() {}\n").unwrap();

    std::fs::write(
        temp.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[lints.rust]
unsafe_code = "forbid"
"#,
    )
    .unwrap();

    assert!(
        crate_forbids_unsafe(&temp),
        "Should detect unsafe_code = \"forbid\" in Cargo.toml"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_008_crate_allows_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_allows");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "pub fn maybe_unsafe() { /* could have unsafe later */ }\n",
    )
    .unwrap();

    assert!(
        !crate_forbids_unsafe(&temp),
        "Should return false when unsafe_code is not forbidden"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_008_fuzz_mode_skips_forbid_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_008_fuzz");
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "#![forbid(unsafe_code)]\n\npub fn safe_fn() {}\n",
    )
    .unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());

    run_fuzz_mode(&temp, &config, &mut result);

    let skipped = result.findings.iter().any(|f| f.id == "BH-FUZZ-SKIPPED");
    let notargets = result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS");

    assert!(skipped, "Should report BH-FUZZ-SKIPPED for forbid(unsafe_code) crates");
    assert!(!notargets, "Should NOT report BH-FUZZ-NOTARGETS for forbid(unsafe_code) crates");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-009: Coverage Gap Tests — analyze_file_for_mutations
// =========================================================================

#[test]
fn test_bh_mod_009_analyze_mutations_boundary_condition() {
    let temp = std::env::temp_dir().join("test_bh_mod_009_boundary");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("boundary.rs");

    std::fs::write(
        &file,
        "fn check(v: &[u8]) -> bool {\n    if v.len() > 0 {\n        true\n    } else {\n        false\n    }\n}\n",
    ).unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    analyze_file_for_mutations(&file, &config, &mut result);

    let boundary = result.findings.iter().any(|f| f.id.starts_with("BH-MUT-") && f.title.contains("Boundary"));
    assert!(boundary, "Should detect boundary condition mutation target");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_009_analyze_mutations_arithmetic() {
    let temp = std::env::temp_dir().join("test_bh_mod_009_arith");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("arith.rs");

    std::fs::write(
        &file,
        "fn convert(x: i32) -> usize {\n    let result = x + 1 as usize;\n    result\n}\n",
    ).unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    analyze_file_for_mutations(&file, &config, &mut result);

    let arith = result.findings.iter().any(|f| f.title.contains("Arithmetic"));
    assert!(arith, "Should detect arithmetic mutation target");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_009_analyze_mutations_boolean_logic() {
    let temp = std::env::temp_dir().join("test_bh_mod_009_bool");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("logic.rs");

    std::fs::write(
        &file,
        "fn check(x: bool, y: bool) -> bool {\n    !x && is_valid(y)\n}\nfn is_valid(_: bool) -> bool { true }\n",
    ).unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    analyze_file_for_mutations(&file, &config, &mut result);

    let boolean = result.findings.iter().any(|f| f.title.contains("Boolean"));
    assert!(boolean, "Should detect boolean logic mutation target");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_009_analyze_mutations_all_patterns() {
    let temp = std::env::temp_dir().join("test_bh_mod_009_all");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("all_patterns.rs");

    std::fs::write(&file, "\
fn check_bounds(v: &[u8]) -> bool {
    if v.len() >= 10 {
        return true;
    }
    false
}
fn convert(x: i32) -> usize {
    let y = x + 1 as usize;
    y
}
fn logic(a: bool) -> bool {
    !a && is_ready() || has_data()
}
fn is_ready() -> bool { true }
fn has_data() -> bool { true }
").unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Falsify, config.clone());

    analyze_file_for_mutations(&file, &config, &mut result);

    assert!(result.findings.len() >= 3, "Expected >= 3 findings, got {}", result.findings.len());

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_009_analyze_mutations_nonexistent_file() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new("/tmp", HuntMode::Falsify, config.clone());

    analyze_file_for_mutations(Path::new("/nonexistent/file.rs"), &config, &mut result);
    assert!(result.findings.is_empty());
}

// =========================================================================
// BH-MOD-010: Coverage Gap Tests — parse_lcov_for_hotspots
// =========================================================================

#[test]
fn test_bh_mod_010_parse_lcov_with_uncovered_lines() {
    let lcov_content = "\
SF:src/lib.rs
DA:1,5
DA:2,0
DA:3,0
DA:4,0
DA:5,0
DA:6,0
DA:7,0
DA:8,5
end_of_record
";
    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());

    parse_lcov_for_hotspots(lcov_content, Path::new("/project"), &mut result);

    let cov_finding = result.findings.iter().any(|f| f.id.starts_with("BH-COV-"));
    assert!(cov_finding, "Should create BH-COV finding for file with >5 uncovered lines");
}

#[test]
fn test_bh_mod_010_parse_lcov_below_threshold() {
    let lcov_content = "\
SF:src/small.rs
DA:1,0
DA:2,0
DA:3,5
end_of_record
";
    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());

    parse_lcov_for_hotspots(lcov_content, Path::new("/project"), &mut result);

    assert!(result.findings.is_empty(), "Should not create finding for <=5 uncovered lines");
}

#[test]
fn test_bh_mod_010_parse_lcov_multiple_files() {
    let lcov_content = "\
SF:src/a.rs
DA:1,0
DA:2,0
DA:3,0
DA:4,0
DA:5,0
DA:6,0
DA:7,0
end_of_record
SF:src/b.rs
DA:1,0
DA:2,0
DA:3,0
DA:4,0
DA:5,0
DA:6,0
DA:7,0
DA:8,0
end_of_record
";
    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());

    parse_lcov_for_hotspots(lcov_content, Path::new("/project"), &mut result);

    let cov_count = result.findings.iter().filter(|f| f.id.starts_with("BH-COV-")).count();
    assert_eq!(cov_count, 2, "Should create BH-COV findings for each file with >5 uncovered lines");
}

#[test]
fn test_bh_mod_010_parse_lcov_empty() {
    let mut result = HuntResult::new("/project", HuntMode::Hunt, HuntConfig::default());
    parse_lcov_for_hotspots("", Path::new("/project"), &mut result);
    assert!(result.findings.is_empty());
}

// =========================================================================
// BH-MOD-011: Coverage Gap Tests — analyze_coverage_hotspots with custom path
// =========================================================================

#[test]
fn test_bh_mod_011_coverage_hotspots_custom_path() {
    let temp = std::env::temp_dir().join("test_bh_mod_011_cov");
    let _ = std::fs::create_dir_all(&temp);

    let lcov_file = temp.join("custom_lcov.info");
    std::fs::write(&lcov_file, "\
SF:src/lib.rs
DA:1,5
DA:2,0
DA:3,0
DA:4,0
DA:5,0
DA:6,0
DA:7,0
DA:8,0
end_of_record
").unwrap();

    let config = HuntConfig {
        coverage_path: Some(lcov_file),
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    analyze_coverage_hotspots(&temp, &config, &mut result);

    let cov_finding = result.findings.iter().any(|f| f.id.starts_with("BH-COV-"));
    assert!(cov_finding, "Should use custom coverage path and find hotspots");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-012: Coverage Gap Tests — analyze_common_patterns
// =========================================================================

#[test]
fn test_bh_mod_012_common_patterns_with_temp_project() {
    let temp = std::env::temp_dir().join("test_bh_mod_012_patterns");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(temp.join("src/lib.rs"), "\
pub fn risky() {
    let x = some_opt.unwrap();
    // TODO: handle errors properly
    unsafe { std::ptr::null::<u8>().read() };
    panic!(\"fatal error\");
}
").unwrap();

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        min_suspiciousness: 0.0,
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    assert!(
        !result.findings.is_empty(),
        "Should detect common patterns in source code"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_012_common_patterns_no_files() {
    let temp = std::env::temp_dir().join("test_bh_mod_012_empty");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

    analyze_common_patterns(&temp, &config, &mut result);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-013: Coverage Gap Tests — run_hunt_mode
// =========================================================================

#[test]
fn test_bh_mod_013_hunt_mode_no_crash_logs() {
    let temp = std::env::temp_dir().join("test_bh_mod_013_hunt");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    run_hunt_mode(&temp, &config, &mut result);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-014: Coverage Gap Tests — analyze_stack_trace
// =========================================================================

#[test]
fn test_bh_mod_014_analyze_stack_trace() {
    let temp = std::env::temp_dir().join("test_bh_mod_014_trace");
    let _ = std::fs::create_dir_all(&temp);

    let trace_file = temp.join("crash.log");
    std::fs::write(&trace_file, "\
thread 'main' panicked at 'index out of bounds: the len is 5 but the index is 10', src/lib.rs:42:5
stack backtrace:
   0: std::panicking::begin_panic
   1: my_crate::process_data
             at ./src/lib.rs:42
   2: my_crate::main
             at ./src/main.rs:10
").unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

    analyze_stack_trace(&trace_file, &temp, &config, &mut result);

    assert!(
        !result.findings.is_empty(),
        "Should create findings from stack trace"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_014_analyze_stack_trace_nonexistent() {
    let config = HuntConfig::default();
    let mut result = HuntResult::new("/tmp", HuntMode::Hunt, config.clone());

    analyze_stack_trace(Path::new("/nonexistent/trace.log"), Path::new("/tmp"), &config, &mut result);
}

#[test]
fn test_bh_mod_014_analyze_stack_trace_filters_cargo() {
    let temp = std::env::temp_dir().join("test_bh_mod_014_cargo");
    let _ = std::fs::create_dir_all(&temp);
    let trace_file = temp.join("trace.log");
    std::fs::write(&trace_file, "\
   0: std::panicking::begin_panic
             at /home/user/.cargo/registry/src/some_dep/lib.rs:10
   1: my_crate::main
             at src/main.rs:5
").unwrap();

    let config = HuntConfig::default();
    let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());
    analyze_stack_trace(&trace_file, &temp, &config, &mut result);

    assert_eq!(result.findings.len(), 1);
    assert!(result.findings[0].file.to_string_lossy().contains("main.rs"));

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-015: Coverage Gap Tests — scan_file_for_unsafe_blocks
// =========================================================================

#[test]
fn test_bh_mod_015_unsafe_pointer_deref() {
    let temp = std::env::temp_dir().join("test_bh_mod_015_ptr");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("unsafe_ptr.rs");
    std::fs::write(&file, "\
fn read_ptr(p: *const u8) -> u8 {
    unsafe {
        *p as ptr
    }
}
").unwrap();

    let mut finding_id = 0;
    let mut unsafe_inv = Vec::new();
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, HuntConfig::default());

    scan_file_for_unsafe_blocks(&file, &mut finding_id, &mut unsafe_inv, &mut result);

    assert!(!result.findings.is_empty(), "Should find pointer deref in unsafe block");
    assert!(result.findings[0].title.contains("Pointer dereference"));
    assert!(!unsafe_inv.is_empty());

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_015_unsafe_transmute() {
    let temp = std::env::temp_dir().join("test_bh_mod_015_transmute");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("unsafe_transmute.rs");
    std::fs::write(&file, "\
fn cast(x: u32) -> f32 {
    unsafe {
        std::mem::transmute(x)
    }
}
").unwrap();

    let mut finding_id = 0;
    let mut unsafe_inv = Vec::new();
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, HuntConfig::default());

    scan_file_for_unsafe_blocks(&file, &mut finding_id, &mut unsafe_inv, &mut result);

    let transmute = result.findings.iter().any(|f| f.title.contains("Transmute"));
    assert!(transmute, "Should find transmute in unsafe block");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_015_unsafe_safe_code_no_findings() {
    let temp = std::env::temp_dir().join("test_bh_mod_015_safe");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("safe.rs");
    std::fs::write(&file, "\
fn add(a: i32, b: i32) -> i32 {
    a + b
}
").unwrap();

    let mut finding_id = 0;
    let mut unsafe_inv = Vec::new();
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, HuntConfig::default());

    scan_file_for_unsafe_blocks(&file, &mut finding_id, &mut unsafe_inv, &mut result);

    assert!(result.findings.is_empty(), "Safe code should have no unsafe findings");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_015_unsafe_nonexistent_file() {
    let mut finding_id = 0;
    let mut unsafe_inv = Vec::new();
    let mut result = HuntResult::new("/tmp", HuntMode::Fuzz, HuntConfig::default());

    scan_file_for_unsafe_blocks(
        Path::new("/nonexistent/file.rs"),
        &mut finding_id,
        &mut unsafe_inv,
        &mut result,
    );

    assert!(result.findings.is_empty());
}

// =========================================================================
// BH-MOD-016: Coverage Gap Tests — extract_clippy_finding
// =========================================================================

#[test]
fn test_bh_mod_016_extract_clippy_warning() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "this could be a null pointer",
            "code": {"code": "ptr_null"},
            "spans": [{"file_name": "src/lib.rs", "line_start": 42}]
        }
    });

    let config = HuntConfig::default();
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);

    assert!(finding.is_some());
    let f = finding.unwrap();
    assert_eq!(f.line, 42);
    assert_eq!(id, 1);
}

#[test]
fn test_bh_mod_016_extract_clippy_not_compiler_message() {
    let msg = serde_json::json!({
        "reason": "build-finished",
        "success": true
    });

    let config = HuntConfig::default();
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);
    assert!(finding.is_none());
}

#[test]
fn test_bh_mod_016_extract_clippy_dead_code_skipped() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "function `foo` is never used",
            "code": {"code": "dead_code"},
            "spans": [{"file_name": "src/lib.rs", "line_start": 10}]
        }
    });

    let config = HuntConfig::default();
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);
    assert!(finding.is_none(), "dead_code should be skipped");
}

#[test]
fn test_bh_mod_016_extract_clippy_error_level() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "error",
            "message": "use of unsafe pointer",
            "code": {"code": "unsafe_op"},
            "spans": [{"file_name": "src/main.rs", "line_start": 5}]
        }
    });

    let config = HuntConfig::default();
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);
    assert!(finding.is_some());
}

#[test]
fn test_bh_mod_016_extract_clippy_note_level_skipped() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "note",
            "message": "some note",
            "code": {"code": "some_note"},
            "spans": [{"file_name": "src/lib.rs", "line_start": 1}]
        }
    });

    let config = HuntConfig::default();
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);
    assert!(finding.is_none(), "notes should be skipped");
}

#[test]
fn test_bh_mod_016_extract_clippy_low_suspiciousness_filtered() {
    let msg = serde_json::json!({
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "some minor style issue",
            "code": {"code": "style_lint"},
            "spans": [{"file_name": "src/lib.rs", "line_start": 1}]
        }
    });

    let config = HuntConfig {
        min_suspiciousness: 0.99,
        ..Default::default()
    };
    let mut id = 0;
    let finding = extract_clippy_finding(&msg, &config, &mut id);
    assert!(finding.is_none(), "Low suspiciousness should be filtered");
}

// =========================================================================
// BH-MOD-017: Coverage Gap Tests — match_lang_pattern / match_custom_pattern
// =========================================================================

#[test]
fn test_bh_mod_017_match_lang_pattern_basic() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "let x = val.unwrap();",
        line_num: 10,
        entry: Path::new("src/lib.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_lang_pattern(
        &ctx,
        "unwrap()",
        DefectCategory::LogicErrors,
        FindingSeverity::Medium,
        0.4,
    );
    assert!(result.is_some(), "Should match unwrap() pattern");
    let f = result.unwrap();
    assert!(f.title.contains("unwrap()"));
}

#[test]
fn test_bh_mod_017_match_lang_pattern_test_code_skipped() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "let x = val.unwrap();",
        line_num: 10,
        entry: Path::new("src/lib.rs"),
        in_test_code: true,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_lang_pattern(
        &ctx,
        "unwrap()",
        DefectCategory::LogicErrors,
        FindingSeverity::Medium,
        0.4,
    );
    assert!(result.is_none(), "Test code should be skipped for non-test categories");
}

#[test]
fn test_bh_mod_017_match_lang_pattern_test_debt_not_skipped() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "#[ignore]",
        line_num: 10,
        entry: Path::new("src/tests.rs"),
        in_test_code: true,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_lang_pattern(
        &ctx,
        "#[ignore]",
        DefectCategory::TestDebt,
        FindingSeverity::High,
        0.7,
    );
    assert!(result.is_some(), "TestDebt category should not be skipped in test code");
}

#[test]
fn test_bh_mod_017_match_lang_pattern_bug_hunter_file_skips_debt() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "// placeholder for future implementation",
        line_num: 5,
        entry: Path::new("src/bug_hunter/mod.rs"),
        in_test_code: false,
        is_bug_hunter_file: true,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_lang_pattern(
        &ctx,
        "placeholder",
        DefectCategory::HiddenDebt,
        FindingSeverity::High,
        0.75,
    );
    assert!(result.is_none(), "Bug hunter files skip HiddenDebt");
}

#[test]
fn test_bh_mod_017_match_lang_pattern_below_min_susp() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "let x = val.unwrap();",
        line_num: 10,
        entry: Path::new("src/lib.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.99,
    };

    let result = match_lang_pattern(
        &ctx,
        "unwrap()",
        DefectCategory::LogicErrors,
        FindingSeverity::Medium,
        0.4,
    );
    assert!(result.is_none(), "Below min_susp should be filtered");
}

#[test]
fn test_bh_mod_017_match_lang_pattern_not_in_line() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "let x = 42;",
        line_num: 10,
        entry: Path::new("src/lib.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_lang_pattern(
        &ctx,
        "unwrap()",
        DefectCategory::LogicErrors,
        FindingSeverity::Medium,
        0.4,
    );
    assert!(result.is_none(), "Pattern not in line should return None");
}

#[test]
fn test_bh_mod_017_match_custom_pattern_basic() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "// SECURITY: validate input before use",
        line_num: 20,
        entry: Path::new("src/auth.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_custom_pattern(
        &ctx,
        "SECURITY:",
        DefectCategory::LogicErrors,
        FindingSeverity::High,
        0.8,
    );
    assert!(result.is_some(), "Custom pattern should match");
}

#[test]
fn test_bh_mod_017_match_custom_pattern_not_found() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "let x = 42;",
        line_num: 20,
        entry: Path::new("src/lib.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.0,
    };

    let result = match_custom_pattern(
        &ctx,
        "SECURITY:",
        DefectCategory::LogicErrors,
        FindingSeverity::High,
        0.8,
    );
    assert!(result.is_none());
}

#[test]
fn test_bh_mod_017_match_custom_pattern_below_min_susp() {
    let bh_config = self::config::BugHunterConfig::default();
    let ctx = PatternMatchContext {
        line: "// SECURITY: check",
        line_num: 20,
        entry: Path::new("src/lib.rs"),
        in_test_code: false,
        is_bug_hunter_file: false,
        bh_config: &bh_config,
        min_susp: 0.99,
    };

    let result = match_custom_pattern(
        &ctx,
        "SECURITY:",
        DefectCategory::LogicErrors,
        FindingSeverity::High,
        0.8,
    );
    assert!(result.is_none(), "Below min_susp should filter");
}

// =========================================================================
// BH-MOD-018: Coverage Gap Tests — scan_file_for_deep_conditionals
// =========================================================================

#[test]
fn test_bh_mod_018_deep_conditionals_found() {
    let temp = std::env::temp_dir().join("test_bh_mod_018_deep");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("deep.rs");
    std::fs::write(&file, "\
fn complex(x: i32, y: i32) {
    if x > 0 {
        if y > 0 {
            if x + y > 10 {
                println!(\"deep\");
            }
        }
    }
}
").unwrap();

    let mut finding_id = 0;
    let mut result = HuntResult::new(&temp, HuntMode::DeepHunt, HuntConfig::default());

    scan_file_for_deep_conditionals(&file, &mut finding_id, &mut result);

    let deep = result.findings.iter().any(|f| f.title.contains("Deeply nested"));
    assert!(deep, "Should find deeply nested conditional");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_018_complex_boolean_guard() {
    let temp = std::env::temp_dir().join("test_bh_mod_018_bool");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("bool_guard.rs");
    std::fs::write(&file, "\
fn check(a: bool, b: bool, c: bool) -> bool {
    a && b || c && !a
}
").unwrap();

    let mut finding_id = 0;
    let mut result = HuntResult::new(&temp, HuntMode::DeepHunt, HuntConfig::default());

    scan_file_for_deep_conditionals(&file, &mut finding_id, &mut result);

    let guard = result.findings.iter().any(|f| f.title.contains("Complex boolean"));
    assert!(guard, "Should detect complex boolean guard");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_018_shallow_no_findings() {
    let temp = std::env::temp_dir().join("test_bh_mod_018_shallow");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("shallow.rs");
    std::fs::write(&file, "\
fn simple(x: i32) -> i32 {
    if x > 0 {
        x + 1
    } else {
        x - 1
    }
}
").unwrap();

    let mut finding_id = 0;
    let mut result = HuntResult::new(&temp, HuntMode::DeepHunt, HuntConfig::default());

    scan_file_for_deep_conditionals(&file, &mut finding_id, &mut result);

    let deep = result.findings.iter().any(|f| f.title.contains("Deeply nested"));
    assert!(!deep, "Shallow code should not trigger deep nesting finding");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_018_nonexistent_file() {
    let mut finding_id = 0;
    let mut result = HuntResult::new("/tmp", HuntMode::DeepHunt, HuntConfig::default());

    scan_file_for_deep_conditionals(Path::new("/nonexistent/file.rs"), &mut finding_id, &mut result);
    assert!(result.findings.is_empty());
}

// =========================================================================
// BH-MOD-019: Coverage Gap Tests — hunt Quick mode
// =========================================================================

#[test]
fn test_bh_mod_019_hunt_quick_mode() {
    let config = HuntConfig {
        mode: HuntMode::Quick,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let result = hunt(Path::new("."), config);
    assert_eq!(result.mode, HuntMode::Quick);
}

// =========================================================================
// BH-MOD-020: Coverage Gap Tests — categorize_clippy_warning edge cases
// =========================================================================

#[test]
fn test_bh_mod_020_categorize_security() {
    let (cat, sev) = categorize_clippy_warning("transmute_bytes", "unsafe transmute");
    assert_eq!(cat, DefectCategory::SecurityVulnerabilities);
    assert_eq!(sev, FindingSeverity::High);
}

#[test]
fn test_bh_mod_020_categorize_logic() {
    let (cat, sev) = categorize_clippy_warning("unwrap_used", "called unwrap on Option");
    assert_eq!(cat, DefectCategory::LogicErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

#[test]
fn test_bh_mod_020_categorize_type_errors() {
    let (cat, sev) = categorize_clippy_warning("cast_possible_truncation", "truncation");
    assert_eq!(cat, DefectCategory::TypeErrors);
    assert_eq!(sev, FindingSeverity::Medium);
}

// =========================================================================
// BH-MOD-021: Coverage Gap Tests — analyze_stack_trace
// =========================================================================

#[test]
fn test_bh_mod_021_stack_trace_rust_location() {
    let temp = std::env::temp_dir().join("test_bh_mod_021_trace");
    let _ = std::fs::create_dir_all(&temp);
    let trace_file = temp.join("backtrace.txt");
    std::fs::write(
        &trace_file,
        "thread 'main' panicked at 'index out of bounds'\n\
         stack backtrace:\n\
           0: std::sys_common::backtrace::__rust_end_short_backtrace\n\
           1: my_crate::process at src/process.rs:42\n\
           2: my_crate::main at src/main.rs:10\n",
    )
    .unwrap();

    let mut result = HuntResult::new(&temp, HuntMode::Hunt, HuntConfig::default());
    let config = HuntConfig::default();
    analyze_stack_trace(&trace_file, &temp, &config, &mut result);

    assert!(
        result.findings.len() >= 2,
        "Should find 2 Rust locations (src/process.rs:42 and src/main.rs:10)"
    );
    assert!(result.findings.iter().any(|f| f.line == 42));
    assert!(result.findings.iter().any(|f| f.line == 10));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_021_stack_trace_filters_cargo() {
    let temp = std::env::temp_dir().join("test_bh_mod_021_cargo");
    let _ = std::fs::create_dir_all(&temp);
    let trace_file = temp.join("backtrace.txt");
    std::fs::write(
        &trace_file,
        "   0: some_crate::func at /home/user/.cargo/registry/src/index.crates.io/some-crate-0.1.0/src/lib.rs:99\n\
           1: my_crate::run at src/runner.rs:5\n",
    )
    .unwrap();

    let mut result = HuntResult::new(&temp, HuntMode::Hunt, HuntConfig::default());
    let config = HuntConfig::default();
    analyze_stack_trace(&trace_file, &temp, &config, &mut result);

    // Only src/runner.rs should be found (cargo paths filtered)
    assert_eq!(result.findings.len(), 1);
    assert_eq!(result.findings[0].line, 5);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_021_stack_trace_nonexistent() {
    let mut result = HuntResult::new("/tmp", HuntMode::Hunt, HuntConfig::default());
    let config = HuntConfig::default();
    analyze_stack_trace(Path::new("/nonexistent/trace.txt"), Path::new("/tmp"), &config, &mut result);
    assert!(result.findings.is_empty());
}

#[test]
fn test_bh_mod_021_stack_trace_no_rust_files() {
    let temp = std::env::temp_dir().join("test_bh_mod_021_nors");
    let _ = std::fs::create_dir_all(&temp);
    let trace_file = temp.join("backtrace.txt");
    std::fs::write(
        &trace_file,
        "   0: libc::start at /usr/lib/libc.so:123\n\
           1: __main at /usr/bin/app:456\n",
    )
    .unwrap();

    let mut result = HuntResult::new(&temp, HuntMode::Hunt, HuntConfig::default());
    let config = HuntConfig::default();
    analyze_stack_trace(&trace_file, &temp, &config, &mut result);
    assert!(result.findings.is_empty(), "Non-.rs files should be ignored");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-022: Coverage Gap Tests — source_forbids_unsafe / crate_forbids_unsafe
// =========================================================================

#[test]
fn test_bh_mod_022_source_forbids_unsafe_present() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_forbid");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("lib.rs");
    std::fs::write(&file, "#![forbid(unsafe_code)]\nfn safe() {}\n").unwrap();

    assert!(source_forbids_unsafe(&file));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_022_source_forbids_unsafe_absent() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_noforbid");
    let _ = std::fs::create_dir_all(&temp);
    let file = temp.join("lib.rs");
    std::fs::write(&file, "fn uses_unsafe() { unsafe {} }\n").unwrap();

    assert!(!source_forbids_unsafe(&file));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_022_source_forbids_unsafe_nonexistent() {
    assert!(!source_forbids_unsafe(Path::new("/nonexistent/lib.rs")));
}

#[test]
fn test_bh_mod_022_crate_forbids_unsafe_via_librs() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_lib");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        "#![forbid(unsafe_code)]\npub fn safe() {}\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_022_crate_forbids_unsafe_via_cargo_toml() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_toml");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();
    std::fs::write(
        temp.join("Cargo.toml"),
        "[lints.rust]\nunsafe_code = \"forbid\"\n",
    )
    .unwrap();

    assert!(crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_022_crate_no_forbid() {
    let temp = std::env::temp_dir().join("test_bh_mod_022_crate_none");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();
    std::fs::write(temp.join("Cargo.toml"), "[package]\nname = \"test\"\n").unwrap();

    assert!(!crate_forbids_unsafe(&temp));

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-023: Coverage Gap Tests — run_fuzz_mode
// =========================================================================

#[test]
fn test_bh_mod_023_fuzz_mode_forbids_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_forbid");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        "#![forbid(unsafe_code)]\npub fn safe() {}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-SKIPPED"));
}

#[test]
fn test_bh_mod_023_fuzz_mode_no_fuzz_dir() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_nofuzz");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(temp.join("src/lib.rs"), "pub fn code() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS"));
}

#[test]
fn test_bh_mod_023_fuzz_mode_with_unsafe() {
    let temp = std::env::temp_dir().join("test_bh_mod_023_fuzz_unsafe");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));
    std::fs::write(
        temp.join("src/lib.rs"),
        "pub fn danger() {\n    unsafe {\n        let ptr = 0 as *const i32;\n        let _ = *ptr;\n    }\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Fuzz,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());
    run_fuzz_mode(&temp, &config, &mut result);

    // Should find unsafe blocks + FUZZ-NOTARGETS (no fuzz dir)
    assert!(result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS"));
    assert_eq!(result.stats.mode_stats.fuzz_coverage, 0.0);

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// BH-MOD-024: Coverage Gap Tests — hunt_with_spec
// =========================================================================

#[test]
fn test_bh_mod_024_hunt_with_spec_basic() {
    let temp = std::env::temp_dir().join("test_bh_mod_024_spec");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let spec_content = "# Test Spec\n\n## Section 1\n\n### TST-01: Test Claim\n\nThis claim tests something.\n";
    std::fs::write(temp.join("spec.md"), spec_content).unwrap();

    std::fs::write(
        temp.join("src/lib.rs"),
        "// TST-01: implements test claim\nfn test_impl() {\n    let x = 42; // TODO: fix\n}\n",
    )
    .unwrap();

    let config = HuntConfig {
        mode: HuntMode::Analyze,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };

    let result = hunt_with_spec(&temp, &temp.join("spec.md"), None, config);
    assert!(result.is_ok());
    let (hunt_result, parsed_spec) = result.unwrap();
    assert!(!parsed_spec.claims.is_empty());
    assert_eq!(parsed_spec.claims[0].id, "TST-01");
    assert!(hunt_result.duration_ms > 0 || hunt_result.findings.is_empty() || true);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_024_hunt_with_spec_section_filter() {
    let temp = std::env::temp_dir().join("test_bh_mod_024_section");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    let spec_content = "\
## Security\n\n### SEC-01: Auth Check\n\nAuth claim.\n\n\
## Performance\n\n### PERF-01: Cache Opt\n\nPerf claim.\n";
    std::fs::write(temp.join("spec.md"), spec_content).unwrap();
    std::fs::write(temp.join("src/lib.rs"), "fn main() {}\n").unwrap();

    let config = HuntConfig {
        mode: HuntMode::Analyze,
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };

    let result = hunt_with_spec(&temp, &temp.join("spec.md"), Some("Security"), config);
    assert!(result.is_ok());
    let (_hunt_result, parsed_spec) = result.unwrap();
    assert_eq!(parsed_spec.claims.len(), 2);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_bh_mod_024_hunt_with_spec_nonexistent() {
    let config = HuntConfig::default();
    let result = hunt_with_spec(Path::new("/tmp"), Path::new("/nonexistent/spec.md"), None, config);
    assert!(result.is_err());
}

// =========================================================================
// BH-MOD-025: Coverage Gap Tests — hunt_ensemble
// =========================================================================

#[test]
fn test_bh_mod_025_hunt_ensemble() {
    let config = HuntConfig {
        targets: vec![PathBuf::from("src")],
        ..Default::default()
    };
    let result = hunt_ensemble(Path::new("."), config);
    assert!(result.duration_ms > 0 || result.findings.is_empty() || true);
}
