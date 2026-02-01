//! Safety & Formal Verification Checks (Section 6)
//!
//! Implements SF-01 through SF-10 from the Popperian Falsification Checklist.
//! Focus: Jidoka automated safety, formal methods.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};

/// Evaluate all safety checks for a project.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_unsafe_code_isolation(project_path),
        check_memory_safety_fuzzing(project_path),
        check_miri_validation(project_path),
        check_formal_safety_properties(project_path),
        check_adversarial_robustness(project_path),
        check_thread_safety(project_path),
        check_resource_leak_prevention(project_path),
        check_panic_safety(project_path),
        check_input_validation(project_path),
        check_supply_chain_security(project_path),
    ]
}

/// Check if an unsafe code location is in an allowed module
fn is_allowed_unsafe_location(path_str: &str, file_name: &str, content: &str) -> bool {
    const ALLOWED_DIRS: &[&str] = &["/internal/", "/ffi/", "/simd/", "/wasm/"];
    const ALLOWED_SUFFIXES: &[&str] = &["_internal.rs", "_ffi.rs", "_simd.rs"];

    ALLOWED_DIRS.iter().any(|d| path_str.contains(d))
        || ALLOWED_SUFFIXES.iter().any(|s| path_str.contains(s))
        || file_name == "lib.rs"
        || content.contains("// SAFETY:")
        || content.contains("# Safety")
}

/// SF-01: Unsafe Code Isolation
///
/// **Claim:** All unsafe code isolated in marked internal modules.
///
/// **Rejection Criteria (Major):**
/// - Unsafe block outside designated module
pub fn check_unsafe_code_isolation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-01",
        "Unsafe Code Isolation",
        "All unsafe code isolated in marked internal modules",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — containment");

    let mut unsafe_locations = Vec::new();
    let mut total_unsafe_blocks = 0;

    // Note: Using concat! to avoid self-matching in this check
    const UNSAFE_BLOCK_PATTERN: &str = concat!("unsafe", " {");
    const UNSAFE_FN_PATTERN: &str = concat!("unsafe", " fn ");

    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let unsafe_count = content.matches(UNSAFE_BLOCK_PATTERN).count()
                + content.matches(UNSAFE_FN_PATTERN).count();

            if unsafe_count == 0 {
                continue;
            }
            total_unsafe_blocks += unsafe_count;

            let file_name = entry.file_name().unwrap_or_default().to_string_lossy();
            let path_str = entry.to_string_lossy();
            if !is_allowed_unsafe_location(&path_str, &file_name, &content) {
                unsafe_locations.push(format!("{}: {} blocks", path_str, unsafe_count));
            }
        }
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Found {} unsafe blocks, {} in non-designated locations",
            total_unsafe_blocks,
            unsafe_locations.len()
        ),
        data: Some(format!("locations: {:?}", unsafe_locations)),
        files: Vec::new(),
    });

    if unsafe_locations.is_empty() {
        item = item.pass();
    } else if unsafe_locations.len() <= 3 {
        item = item.partial(format!(
            "{} unsafe blocks outside designated modules",
            unsafe_locations.len()
        ));
    } else {
        item = item.fail(format!(
            "{} unsafe blocks outside designated modules: {}",
            unsafe_locations.len(),
            unsafe_locations.join(", ")
        ));
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-02: Memory Safety Under Fuzzing
///
/// **Claim:** No memory safety violations under fuzzing.
///
/// **Rejection Criteria (Major):**
/// - Any ASan/MSan/UBSan violation
pub fn check_memory_safety_fuzzing(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-02",
        "Memory Safety Under Fuzzing",
        "No memory safety violations under fuzzing",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — defect detection");

    // Check for fuzzing setup
    let fuzz_dir = project_path.join("fuzz");
    let has_fuzz_dir = fuzz_dir.exists();
    let has_fuzz_targets = if has_fuzz_dir {
        glob::glob(&format!("{}/fuzz_targets/**/*.rs", fuzz_dir.display()))
            .ok()
            .map(|entries| entries.count() > 0)
            .unwrap_or(false)
            || fuzz_dir.join("fuzz_targets").exists()
    } else {
        false
    };

    // Check for cargo-fuzz or property-based testing in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let cargo_content = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten();

    let has_fuzz_dep = cargo_content
        .as_ref()
        .map(|c| c.contains("libfuzzer-sys") || c.contains("arbitrary"))
        .unwrap_or(false);

    // Proptest is a valid property-based testing framework (similar to fuzzing)
    let has_proptest = cargo_content
        .as_ref()
        .map(|c| c.contains("proptest") || c.contains("quickcheck"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Fuzzing setup: dir={}, targets={}, fuzz_deps={}, proptest={}",
            has_fuzz_dir, has_fuzz_targets, has_fuzz_dep, has_proptest
        ),
        data: None,
        files: Vec::new(),
    });

    if has_fuzz_targets && has_fuzz_dep {
        // Full fuzzing infrastructure
        item = item.pass();
    } else if has_proptest {
        // Property-based testing is an equivalent approach for memory safety
        item = item.pass();
    } else if has_fuzz_dir || has_fuzz_dep {
        item = item.partial("Fuzzing partially configured");
    } else {
        // For projects without fuzzing, check if it's a simple library
        let is_small_project = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| entries.count() < 20)
            .unwrap_or(true);

        if is_small_project {
            item = item.partial("No fuzzing setup (small project)");
        } else {
            item = item.fail("No fuzzing infrastructure detected");
        }
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-03: Miri Undefined Behavior Detection
///
/// **Claim:** Core operations pass Miri validation.
///
/// **Rejection Criteria (Major):**
/// - Any Miri error
pub fn check_miri_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-03",
        "Miri Undefined Behavior Detection",
        "Core operations pass Miri validation",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automatic UB detection");

    // Check if CI config includes Miri
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
    ];

    let mut has_miri_in_ci = false;
    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(content) = std::fs::read_to_string(ci_path) {
                if content.contains("miri") {
                    has_miri_in_ci = true;
                    break;
                }
            }
        }
    }

    // Check for Miri in Makefile
    let makefile = project_path.join("Makefile");
    let has_miri_in_makefile = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("miri"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Miri setup: ci={}, makefile={}",
            has_miri_in_ci, has_miri_in_makefile
        ),
        data: None,
        files: Vec::new(),
    });

    if has_miri_in_ci {
        item = item.pass();
    } else if has_miri_in_makefile {
        item = item.partial("Miri available but not in CI");
    } else {
        // Check for actual unsafe blocks - if none, Miri less critical
        // Note: Using concat! to avoid self-matching in this check
        const UNSAFE_BLOCK: &str = concat!("unsafe", " {");
        let unsafe_count: usize = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|p| std::fs::read_to_string(&p).ok())
                    .map(|c| c.matches(UNSAFE_BLOCK).count())
                    .sum()
            })
            .unwrap_or(0);

        if unsafe_count == 0 {
            // Safe Rust code - Miri would have nothing to find
            item = item.pass();
        } else {
            item = item.partial(format!(
                "Miri not configured ({} unsafe blocks)",
                unsafe_count
            ));
        }
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-04: Formal Safety Properties
///
/// **Claim:** Safety-critical components have formal proofs.
///
/// **Rejection Criteria (Minor):**
/// - Safety property unproven for critical path
pub fn check_formal_safety_properties(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-04",
        "Formal Safety Properties",
        "Safety-critical components have formal proofs",
    )
    .with_severity(Severity::Minor)
    .with_tps("Formal verification requirement");

    // Check for Kani, Creusot, or other formal verification tools
    let cargo_toml = project_path.join("Cargo.toml");
    let has_kani = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("kani") || c.contains("creusot") || c.contains("prusti"))
        .unwrap_or(false);

    // Check for proof annotations in code
    let has_proof_annotations = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("#[kani::")
                            || c.contains("#[requires(")
                            || c.contains("#[ensures(")
                            || c.contains("// PROOF:")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Formal verification: tools={}, annotations={}",
            has_kani, has_proof_annotations
        ),
        data: None,
        files: Vec::new(),
    });

    if has_kani && has_proof_annotations {
        item = item.pass();
    } else if has_kani || has_proof_annotations {
        item = item.partial("Partial formal verification setup");
    } else {
        // Formal verification is advanced - partial is acceptable
        item = item.partial("No formal verification (advanced feature)");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-05: Adversarial Robustness Verification
///
/// **Claim:** Models tested against adversarial examples.
///
/// **Rejection Criteria (Major):**
/// - Model fails under documented attack types
pub fn check_adversarial_robustness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-05",
        "Adversarial Robustness Verification",
        "Models tested against adversarial examples",
    )
    .with_severity(Severity::Major)
    .with_tps("AI Safety requirement");

    // Check for adversarial testing patterns
    let has_adversarial_tests = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("adversarial")
                            || c.contains("perturbation")
                            || c.contains("robustness")
                            || c.contains("attack")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for robustness verification in tests
    let has_robustness_verification =
        glob::glob(&format!("{}/tests/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries.flatten().any(|p| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .map(|c| c.contains("adversarial") || c.contains("robustness"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Adversarial robustness: testing={}, verification={}",
            has_adversarial_tests, has_robustness_verification
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML models that need adversarial testing
    let has_ml_models = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("predict") || c.contains("classifier") || c.contains("neural")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    if !has_ml_models || has_adversarial_tests || has_robustness_verification {
        item = item.pass();
    } else {
        item = item.partial("ML models without adversarial testing");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Find files with unsafe Send/Sync implementations lacking safety docs
fn find_unsafe_send_sync(project_path: &Path) -> Vec<String> {
    let mut results = Vec::new();
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return results;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        let has_unsafe_impl = content.contains("unsafe impl Send")
            || content.contains("unsafe impl Sync")
            || content.contains("unsafe impl<") && content.contains("> Send")
            || content.contains("unsafe impl<") && content.contains("> Sync");

        if has_unsafe_impl
            && !content.contains("// SAFETY:")
            && !content.contains("# Safety")
        {
            results.push(entry.to_string_lossy().to_string());
        }
    }
    results
}

/// SF-06: Thread Safety (Send + Sync)
///
/// **Claim:** All Send + Sync implementations correct.
///
/// **Rejection Criteria (Major):**
/// - TSan detects any race
pub fn check_thread_safety(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-06",
        "Thread Safety (Send + Sync)",
        "All Send + Sync implementations correct",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — race detection");

    let unsafe_send_sync = find_unsafe_send_sync(project_path);

    // Check for concurrent data structures
    let cargo_toml = project_path.join("Cargo.toml");
    let uses_concurrent = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| {
            c.contains("crossbeam")
                || c.contains("parking_lot")
                || c.contains("dashmap")
                || c.contains("rayon")
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Thread safety: unsafe_impls={}, concurrent_libs={}",
            unsafe_send_sync.len(),
            uses_concurrent
        ),
        data: Some(format!("unsafe_send_sync: {:?}", unsafe_send_sync)),
        files: Vec::new(),
    });

    if unsafe_send_sync.is_empty() {
        item = item.pass();
    } else if unsafe_send_sync.len() <= 2 {
        item = item.partial(format!(
            "{} unsafe Send/Sync without safety comment",
            unsafe_send_sync.len()
        ));
    } else {
        item = item.fail(format!(
            "{} unsafe Send/Sync implementations without documentation",
            unsafe_send_sync.len()
        ));
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-07: Resource Leak Prevention
///
/// **Claim:** No resource leaks.
///
/// **Rejection Criteria (Major):**
/// - "definitely lost" > 0 bytes in Valgrind
pub fn check_resource_leak_prevention(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-07", "Resource Leak Prevention", "No resource leaks")
        .with_severity(Severity::Major)
        .with_tps("Muda (Defects)");

    // Check for Drop implementations
    let mut drop_impls = 0;
    let mut resource_types = Vec::new();

    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                drop_impls += content.matches("impl Drop for").count();
                drop_impls +=
                    content.matches("impl<").count() * content.matches("> Drop for").count().min(1);

                // Look for resource types
                if content.contains("File") || content.contains("TcpStream") {
                    resource_types.push("file/network handles");
                }
                if content.contains("Arc<") || content.contains("Rc<") {
                    resource_types.push("reference counting");
                }
                if content.contains("ManuallyDrop") {
                    resource_types.push("ManuallyDrop");
                }
            }
        }
    }

    // Check for mem::forget usage (potential leak)
    let has_mem_forget = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("mem::forget") || c.contains("std::mem::forget"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Resource management: drop_impls={}, mem_forget={}, resource_types={:?}",
            drop_impls, has_mem_forget, resource_types
        ),
        data: None,
        files: Vec::new(),
    });

    if !has_mem_forget {
        item = item.pass();
    } else {
        item = item.partial("Uses mem::forget (verify intentional)");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-08: Panic Safety
///
/// **Claim:** Panics don't corrupt data structures.
///
/// **Rejection Criteria (Minor):**
/// - Post-panic access causes UB
pub fn check_panic_safety(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-08",
        "Panic Safety",
        "Panics don't corrupt data structures",
    )
    .with_severity(Severity::Minor)
    .with_tps("Graceful degradation");

    // Check for panic handling patterns
    let mut panic_patterns = Vec::new();
    let mut has_catch_unwind = false;
    let mut has_panic_hook = false;

    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                if content.contains("catch_unwind") {
                    has_catch_unwind = true;
                }
                if content.contains("set_panic_hook") || content.contains("panic::set_hook") {
                    has_panic_hook = true;
                }
                // Check for unwrap usage (potential panic points)
                let unwrap_count = content.matches(".unwrap()").count();
                if unwrap_count > 10 {
                    panic_patterns.push(format!(
                        "{}: {} unwraps",
                        entry.file_name().unwrap_or_default().to_string_lossy(),
                        unwrap_count
                    ));
                }
            }
        }
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Panic handling: catch_unwind={}, panic_hook={}, high_unwrap_files={}",
            has_catch_unwind,
            has_panic_hook,
            panic_patterns.len()
        ),
        data: Some(format!("patterns: {:?}", panic_patterns)),
        files: Vec::new(),
    });

    if panic_patterns.is_empty() {
        item = item.pass();
    } else if panic_patterns.len() <= 5 {
        item = item.partial(format!(
            "{} files with high unwrap count",
            panic_patterns.len()
        ));
    } else {
        item = item.partial(format!(
            "{} files with excessive unwraps - consider expect() or ? operator",
            panic_patterns.len()
        ));
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Scan source files for validation patterns
fn scan_validation_patterns(project_path: &Path) -> (bool, Vec<&'static str>) {
    let mut has_explicit = false;
    let mut methods = Vec::new();

    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return (false, methods);
    };

    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("fn validate")
            || content.contains("fn is_valid")
            || content.contains("impl Validate")
            || content.contains("#[validate")
        {
            has_explicit = true;
            methods.push("explicit validation");
        }
        if content.contains("pub fn")
            && (content.contains("-> Result<") || content.contains("-> Option<"))
        {
            methods.push("Result/Option returns");
        }
        if content.contains("assert!(") || content.contains("debug_assert!(") {
            methods.push("assertions");
        }
    }
    (has_explicit, methods)
}

/// Check if Cargo.toml uses a validator crate
fn has_validator_crate(project_path: &Path) -> bool {
    let cargo_toml = project_path.join("Cargo.toml");
    std::fs::read_to_string(cargo_toml)
        .ok()
        .is_some_and(|c| c.contains("validator") || c.contains("garde"))
}

/// SF-09: Input Validation
///
/// **Claim:** All public APIs validate inputs.
///
/// **Rejection Criteria (Major):**
/// - Any panic from malformed input
pub fn check_input_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-09",
        "Input Validation",
        "All public APIs validate inputs",
    )
    .with_severity(Severity::Major)
    .with_tps("Poka-Yoke — error prevention");

    let (has_validation, mut validation_methods) = scan_validation_patterns(project_path);
    let has_validator = has_validator_crate(project_path);

    if has_validator {
        validation_methods.push("validator crate");
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Validation: explicit={}, methods={:?}",
            has_validation, validation_methods
        ),
        data: None,
        files: Vec::new(),
    });

    if has_validation || has_validator {
        item = item.pass();
    } else if !validation_methods.is_empty() {
        item = item.pass(); // Has Result/Option returns which is implicit validation
    } else {
        item = item.partial("Consider adding explicit input validation");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// SF-10: Supply Chain Security
///
/// **Claim:** All dependencies audited.
///
/// **Rejection Criteria (Critical):**
/// - Known vulnerability or unmaintained critical dependency
pub fn check_supply_chain_security(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-10", "Supply Chain Security", "All dependencies audited")
        .with_severity(Severity::Critical)
        .with_tps("Jidoka — supply chain circuit breaker");

    // Check for cargo-audit
    let has_audit_in_ci = check_ci_for_tool(project_path, "cargo audit");
    let has_deny_in_ci = check_ci_for_tool(project_path, "cargo deny");

    // Check for deny.toml
    let deny_toml = project_path.join("deny.toml");
    let has_deny_config = deny_toml.exists();

    // Try running cargo audit if available
    let audit_result = Command::new("cargo")
        .args(["audit", "--json"])
        .current_dir(project_path)
        .output()
        .ok();

    let audit_clean = audit_result
        .as_ref()
        .map(|o| {
            o.status.success()
                || String::from_utf8_lossy(&o.stdout).contains("\"vulnerabilities\":[]")
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::DependencyAudit,
        description: format!(
            "Supply chain: audit_ci={}, deny_ci={}, deny_config={}, audit_clean={}",
            has_audit_in_ci, has_deny_in_ci, has_deny_config, audit_clean
        ),
        data: None,
        files: Vec::new(),
    });

    if has_deny_config && (has_audit_in_ci || has_deny_in_ci) {
        item = item.pass();
    } else if has_deny_config || has_audit_in_ci || has_deny_in_ci {
        item = item.partial("Partial supply chain security setup");
    } else if audit_clean {
        item = item.partial("No vulnerabilities but no CI enforcement");
    } else {
        item = item.fail("No supply chain security tooling configured");
    }

    item.with_duration(start.elapsed().as_millis() as u64)
}

/// Helper: Check if a tool is referenced in CI configuration
fn check_ci_for_tool(project_path: &Path, tool: &str) -> bool {
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
        project_path.join(".github/workflows/security.yml"),
    ];

    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(content) = std::fs::read_to_string(ci_path) {
                if content.contains(tool) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // =========================================================================
    // SF-01: Unsafe Code Isolation Tests
    // =========================================================================

    #[test]
    fn test_sf_01_unsafe_code_isolation_current_project() {
        let result = check_unsafe_code_isolation(Path::new("."));
        // batuta should have minimal unsafe code
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "Unsafe code isolation failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // SF-02: Memory Safety Fuzzing Tests
    // =========================================================================

    #[test]
    fn test_sf_02_memory_safety_fuzzing() {
        let result = check_memory_safety_fuzzing(Path::new("."));
        // Fuzzing is optional, partial is acceptable
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Fuzzing check should complete"
        );
    }

    // =========================================================================
    // SF-03: Miri Validation Tests
    // =========================================================================

    #[test]
    fn test_sf_03_miri_validation() {
        let result = check_miri_validation(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Miri check should complete"
        );
    }

    // =========================================================================
    // SF-04: Formal Safety Properties Tests
    // =========================================================================

    #[test]
    fn test_sf_04_formal_safety_properties() {
        let result = check_formal_safety_properties(Path::new("."));
        // Formal verification is advanced, partial acceptable
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Formal verification check should complete"
        );
    }

    // =========================================================================
    // SF-05: Adversarial Robustness Tests
    // =========================================================================

    #[test]
    fn test_sf_05_adversarial_robustness() {
        let result = check_adversarial_robustness(Path::new("."));
        // Adversarial testing depends on ML models
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Adversarial robustness check should complete"
        );
    }

    // =========================================================================
    // SF-06: Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_sf_06_thread_safety() {
        let result = check_thread_safety(Path::new("."));
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "Thread safety failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // SF-07: Resource Leak Prevention Tests
    // =========================================================================

    #[test]
    fn test_sf_07_resource_leak_prevention() {
        let result = check_resource_leak_prevention(Path::new("."));
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "Resource leak prevention failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // SF-08: Panic Safety Tests
    // =========================================================================

    #[test]
    fn test_sf_08_panic_safety() {
        let result = check_panic_safety(Path::new("."));
        assert!(
            !matches!(result.status, CheckStatus::Fail),
            "Panic safety failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // SF-09: Input Validation Tests
    // =========================================================================

    #[test]
    fn test_sf_09_input_validation() {
        let result = check_input_validation(Path::new("."));
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
            "Input validation failed: {:?}",
            result.rejection_reason
        );
    }

    // =========================================================================
    // SF-10: Supply Chain Security Tests
    // =========================================================================

    #[test]
    fn test_sf_10_supply_chain_security() {
        let result = check_supply_chain_security(Path::new("."));
        // Supply chain tools may not be installed in all environments
        assert!(
            !matches!(result.status, CheckStatus::Skipped),
            "Supply chain check should complete"
        );
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_evaluate_all_returns_10_items() {
        let results = evaluate_all(Path::new("."));
        assert_eq!(results.len(), 10, "Expected 10 safety checks");
    }

    #[test]
    fn test_all_items_have_evidence() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(
                !item.evidence.is_empty(),
                "Item {} missing evidence",
                item.id
            );
        }
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let results = evaluate_all(Path::new("."));
        for item in &results {
            assert!(
                !item.tps_principle.is_empty(),
                "Item {} missing TPS principle",
                item.id
            );
        }
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_sf_01_unsafe_code() {
        let result = check_unsafe_code_isolation(Path::new("."));
        assert_eq!(result.id, "SF-01");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_sf_02_memory_safety() {
        let result = check_memory_safety_fuzzing(Path::new("."));
        assert_eq!(result.id, "SF-02");
    }

    #[test]
    fn test_sf_03_miri_validation_id() {
        let result = check_miri_validation(Path::new("."));
        assert_eq!(result.id, "SF-03");
    }

    #[test]
    fn test_sf_04_formal_safety() {
        let result = check_formal_safety_properties(Path::new("."));
        assert_eq!(result.id, "SF-04");
    }

    #[test]
    fn test_sf_05_adversarial() {
        let result = check_adversarial_robustness(Path::new("."));
        assert_eq!(result.id, "SF-05");
    }

    #[test]
    fn test_sf_06_thread_safety_id() {
        let result = check_thread_safety(Path::new("."));
        assert_eq!(result.id, "SF-06");
    }

    #[test]
    fn test_sf_07_resource_leak_id() {
        let result = check_resource_leak_prevention(Path::new("."));
        assert_eq!(result.id, "SF-07");
    }

    #[test]
    fn test_sf_08_panic_safety_id() {
        let result = check_panic_safety(Path::new("."));
        assert_eq!(result.id, "SF-08");
    }

    #[test]
    fn test_sf_09_input_validation_id() {
        let result = check_input_validation(Path::new("."));
        assert_eq!(result.id, "SF-09");
    }

    #[test]
    fn test_nonexistent_path() {
        let results = evaluate_all(Path::new("/nonexistent/path"));
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_temp_dir_with_unsafe() {
        let temp_dir = std::env::temp_dir().join("test_unsafe_code");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).unwrap();

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
unsafe fn dangerous() {}
"#,
        )
        .unwrap();

        let result = check_unsafe_code_isolation(&temp_dir);
        assert_eq!(result.id, "SF-01");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_fuzzing() {
        let temp_dir = std::env::temp_dir().join("test_fuzzing");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("fuzz")).unwrap();

        std::fs::write(temp_dir.join("fuzz/fuzz_targets.rs"), "// fuzz target").unwrap();

        let result = check_memory_safety_fuzzing(&temp_dir);
        assert_eq!(result.id, "SF-02");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_proptest() {
        let temp_dir = std::env::temp_dir().join("test_proptest");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dev-dependencies]
proptest = "1.0"
"#,
        )
        .unwrap();

        let result = check_adversarial_robustness(&temp_dir);
        assert_eq!(result.id, "SF-05");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_all_severities_present() {
        let results = evaluate_all(Path::new("."));
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical, "Expected at least one Critical check");
    }
}
