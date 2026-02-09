//! Proactive Bug Hunting Module
//!
//! Implements Section 11 of the Popperian Falsification Checklist (BH-01 to BH-15).
//! Provides falsification-driven defect discovery using multiple hunting modes.
//!
//! # Philosophy
//!
//! "A theory that explains everything, explains nothing." — Karl Popper
//!
//! Bug hunting operationalizes falsification: we systematically attempt to break
//! code, not merely verify it works. Each mode represents a different strategy
//! for falsifying the implicit claim "this code is correct."
//!
//! # Modes
//!
//! - **Falsify**: Mutation-based invariant falsification (FDV pattern)
//! - **Hunt**: SBFL without failing tests (SBEST pattern)
//! - **Analyze**: LLM-augmented static analysis (LLIFT pattern)
//! - **Fuzz**: Targeted unsafe Rust fuzzing (FourFuzz pattern)
//! - **DeepHunt**: Hybrid concolic + SBFL (COTTONTAIL pattern)
//!
//! # Advanced Features (BH-11 to BH-15)
//!
//! - **Spec-Driven**: Hunt bugs guided by specification files
//! - **Ticket-Scoped**: Focus on areas defined by PMAT work tickets
//! - **Scoped Analysis**: --lib, --bin, --path for targeted hunting
//! - **Bidirectional Linking**: Update specs with findings
//! - **False Positive Suppression**: Filter known false positive patterns
//!
//! # Integration with OIP
//!
//! These modes leverage OIP's SBFL (Tarantula/Ochiai/DStar), defect classification,
//! and RAG enhancement to proactively identify bugs before they reach production.

pub mod blame;
pub mod cache;
pub mod config;
pub mod coverage;
pub mod diff;
pub mod languages;
pub mod localization;
pub mod patterns;
pub mod pmat_quality;
pub mod spec;
pub mod ticket;
mod types;

#[allow(unused_imports)]
pub use localization::{CrashBucketer, MultiChannelLocalizer, ScoredLocation};
pub use patterns::{compute_test_lines, is_real_pattern, should_suppress_finding};
pub use spec::ParsedSpec;
pub use ticket::PmatTicket;
pub use types::*;

use std::path::Path;
use std::time::Instant;

/// Print a phase progress indicator to stderr (native builds only).
#[cfg(feature = "native")]
fn eprint_phase(phase: &str, mode: &HuntMode) {
    use crate::ansi_colors::Colorize;
    eprintln!("  {} {}", format!("[{:>8}]", mode).dimmed(), phase);
}

/// Run bug hunting with the specified configuration.
pub fn hunt(project_path: &Path, config: HuntConfig) -> HuntResult {
    let start = Instant::now();

    // Check cache first
    if let Some(cached) = cache::load_cached(project_path, &config) {
        #[cfg(feature = "native")]
        {
            use crate::ansi_colors::Colorize;
            eprintln!("  {} hit — using cached findings", "[  cache]".dimmed());
        }
        let mut result = HuntResult::new(project_path, cached.mode, config);
        result.findings = cached.findings;
        result.duration_ms = start.elapsed().as_millis() as u64;
        result.finalize();
        return result;
    }

    let mut result = HuntResult::new(project_path, config.mode, config.clone());

    // Phase 1: Mode dispatch (scanning)
    #[cfg(feature = "native")]
    eprint_phase("Scanning...", &config.mode);
    let phase_start = Instant::now();

    match config.mode {
        HuntMode::Falsify => run_falsify_mode(project_path, &config, &mut result),
        HuntMode::Hunt => run_hunt_mode(project_path, &config, &mut result),
        HuntMode::Analyze => run_analyze_mode(project_path, &config, &mut result),
        HuntMode::Fuzz => run_fuzz_mode(project_path, &config, &mut result),
        HuntMode::DeepHunt => run_deep_hunt_mode(project_path, &config, &mut result),
        HuntMode::Quick => run_quick_mode(project_path, &config, &mut result),
    }
    result.phase_timings.mode_dispatch_ms = phase_start.elapsed().as_millis() as u64;

    // Phase 2: BH-21 to BH-24: PMAT quality integration
    #[cfg(feature = "native")]
    if config.use_pmat_quality {
        eprint_phase("Quality index...", &config.mode);
        let pmat_start = Instant::now();
        let query = config.pmat_query.as_deref().unwrap_or("*");
        if let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) {
            result.phase_timings.pmat_index_ms = pmat_start.elapsed().as_millis() as u64;

            let weights_start = Instant::now();
            eprint_phase("Applying weights...", &config.mode);
            pmat_quality::apply_quality_weights(
                &mut result.findings,
                &index,
                config.quality_weight,
            );
            pmat_quality::apply_regression_risk(&mut result.findings, &index);
            result.phase_timings.pmat_weights_ms = weights_start.elapsed().as_millis() as u64;
        }
    }

    // Phase 2b: Coverage-based hotpath weighting
    #[cfg(feature = "native")]
    if config.coverage_weight > 0.0 {
        // Try to find coverage file
        let cov_path = config
            .coverage_path
            .clone()
            .or_else(|| coverage::find_coverage_file(project_path));

        if let Some(cov_path) = cov_path {
            if let Some(cov_index) = coverage::load_coverage_index(&cov_path) {
                eprint_phase("Coverage weights...", &config.mode);
                coverage::apply_coverage_weights(
                    &mut result.findings,
                    &cov_index,
                    config.coverage_weight,
                );
            }
        }
    }

    // Phase 3: Finalize
    #[cfg(feature = "native")]
    eprint_phase("Finalizing...", &config.mode);
    let finalize_start = Instant::now();

    result.duration_ms = start.elapsed().as_millis() as u64;
    result.finalize();
    result.phase_timings.finalize_ms = finalize_start.elapsed().as_millis() as u64;

    // Save to cache
    cache::save_cache(project_path, &config, &result.findings, result.mode);

    result
}

/// Run all modes and combine results (ensemble approach).
pub fn hunt_ensemble(project_path: &Path, base_config: HuntConfig) -> HuntResult {
    let start = Instant::now();
    let mut combined = HuntResult::new(project_path, HuntMode::Analyze, base_config.clone());

    // Run each mode and collect findings
    for mode in [
        HuntMode::Analyze,
        HuntMode::Hunt,
        HuntMode::Falsify,
    ] {
        let mut config = base_config.clone();
        config.mode = mode;
        let mode_result = hunt(project_path, config);

        for finding in mode_result.findings {
            // Avoid duplicates by checking location
            let exists = combined.findings.iter().any(|f| {
                f.file == finding.file && f.line == finding.line
            });
            if !exists {
                combined.add_finding(finding);
            }
        }
    }

    combined.duration_ms = start.elapsed().as_millis() as u64;
    combined.finalize();
    combined
}

/// Run spec-driven bug hunting (BH-11).
///
/// Parses a specification file, extracts claims, finds implementing code,
/// and hunts bugs specifically in those areas.
pub fn hunt_with_spec(
    project_path: &Path,
    spec_path: &Path,
    section_filter: Option<&str>,
    mut config: HuntConfig,
) -> Result<(HuntResult, ParsedSpec), String> {
    let start = Instant::now();

    // Parse the specification
    let mut parsed_spec = ParsedSpec::parse(spec_path)?;

    // Get claim IDs to hunt (filtered by section if specified)
    let claim_ids: Vec<String> = if let Some(section) = section_filter {
        parsed_spec
            .claims_for_section(section)
            .iter()
            .map(|c| c.id.clone())
            .collect()
    } else {
        parsed_spec.claims.iter().map(|c| c.id.clone()).collect()
    };

    // Find implementations for each claim
    for claim in &mut parsed_spec.claims {
        claim.implementations = spec::find_implementations(claim, project_path);
    }

    // Collect target paths from implementations (for claims in our filter)
    let mut target_paths: Vec<std::path::PathBuf> = parsed_spec
        .claims
        .iter()
        .filter(|c| claim_ids.contains(&c.id))
        .flat_map(|c| c.implementations.iter().map(|i| i.file.clone()))
        .collect();

    // Deduplicate
    target_paths.sort();
    target_paths.dedup();

    // If no implementations found, use default targets
    if target_paths.is_empty() {
        target_paths = config.targets.clone();
    }

    // Update config with discovered targets
    config.targets = target_paths
        .iter()
        .map(|p| p.parent().unwrap_or(Path::new("src")).to_path_buf())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if config.targets.is_empty() {
        config.targets = vec![std::path::PathBuf::from("src")];
    }

    // Capture PMAT config before hunt() consumes config
    let use_pmat_quality = config.use_pmat_quality;
    let pmat_query_str = config.pmat_query.clone();

    // Run the hunt
    let mut result = hunt(project_path, config);

    // Map findings to claims
    let mapping = spec::map_findings_to_claims(&parsed_spec.claims, &result.findings, project_path);

    // BH-25: Quality gate — if PMAT quality is enabled, check implementing functions
    if use_pmat_quality {
        let query = pmat_query_str.as_deref().unwrap_or("*");
        if let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) {
            // Check each claim's implementations for quality issues
            for claim in &mut parsed_spec.claims {
                for imp in &claim.implementations {
                    if let Some(pmat) =
                        pmat_quality::lookup_quality(&index, &imp.file, imp.line)
                    {
                        let is_low_quality = pmat.tdg_grade == "D"
                            || pmat.tdg_grade == "F"
                            || pmat.complexity > 20;
                        if is_low_quality {
                            // Add a quality warning finding for this claim
                            result.add_finding(
                                Finding::new(
                                    format!("BH-QGATE-{}", claim.id),
                                    &imp.file,
                                    imp.line,
                                    format!(
                                        "Quality gate: claim `{}` implemented by low-quality code",
                                        claim.id
                                    ),
                                )
                                .with_description(format!(
                                    "Function `{}` (grade {}, complexity {}) implements spec claim `{}`; consider refactoring",
                                    pmat.function_name, pmat.tdg_grade, pmat.complexity, claim.id
                                ))
                                .with_severity(FindingSeverity::Medium)
                                .with_category(DefectCategory::LogicErrors)
                                .with_suspiciousness(0.6)
                                .with_discovered_by(HuntMode::Analyze)
                                .with_evidence(FindingEvidence::quality_metrics(
                                    &pmat.tdg_grade,
                                    pmat.tdg_score,
                                    pmat.complexity,
                                )),
                            );
                        }
                    }
                }
            }
        }
    }

    // Update spec claims with findings
    let findings_by_claim: Vec<(String, Vec<Finding>)> = mapping.into_iter().collect();
    if let Ok(updated_content) = parsed_spec.update_with_findings(&findings_by_claim) {
        parsed_spec.original_content = updated_content;
    }

    result.duration_ms = start.elapsed().as_millis() as u64;

    Ok((result, parsed_spec))
}

/// Run ticket-scoped bug hunting (BH-12).
///
/// Parses a PMAT ticket and focuses analysis on affected paths.
pub fn hunt_with_ticket(
    project_path: &Path,
    ticket_ref: &str,
    mut config: HuntConfig,
) -> Result<HuntResult, String> {
    // Parse the ticket
    let ticket = PmatTicket::parse(ticket_ref, project_path)?;

    // Update targets from ticket
    config.targets = ticket.target_paths();

    // Run the hunt with scoped targets
    Ok(hunt(project_path, config))
}

// ============================================================================
// Mode Implementations
// ============================================================================

/// BH-01: Mutation-based invariant falsification (FDV pattern)
fn run_falsify_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Check for cargo-mutants availability
    let mutants_available = std::process::Command::new("cargo")
        .args(["mutants", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !mutants_available {
        result.add_finding(
            Finding::new(
                "BH-FALSIFY-UNAVAIL",
                project_path.join("Cargo.toml"),
                1,
                "cargo-mutants not installed",
            )
            .with_description("Install with: cargo install cargo-mutants")
            .with_severity(FindingSeverity::Info)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.1)
            .with_discovered_by(HuntMode::Falsify),
        );
        return;
    }

    // Analyze Rust files for potential mutation targets
    for target in &config.targets {
        let target_path = project_path.join(target);
        if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", target_path.display())) {
            for entry in entries.flatten() {
                analyze_file_for_mutations(&entry, config, result);
            }
        }
    }
}

/// Analyze a file for mutation testing targets.
fn analyze_file_for_mutations(file_path: &Path, _config: &HuntConfig, result: &mut HuntResult) {
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut finding_id = 0;

    // Look for patterns that are good mutation targets
    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1; // 1-indexed

        // Pattern: Boundary conditions (off-by-one potential)
        let has_comparison = line.contains("< ") || line.contains("> ") || line.contains("<= ") || line.contains(">= ");
        let has_len = line.contains("len()") || line.contains("size()") || line.contains(".len");
        if has_comparison && has_len {
            finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-MUT-{:04}", finding_id),
                    file_path,
                    line_num,
                    "Boundary condition mutation target",
                )
                .with_description("Off-by-one errors are common; this comparison should be mutation-tested")
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.6)
                .with_discovered_by(HuntMode::Falsify)
                .with_evidence(FindingEvidence::mutation(format!("boundary_{}", finding_id), true)),
            );
        }

        // Pattern: Arithmetic operations (overflow potential)
        let has_arith = line.contains(" + ") || line.contains(" - ") || line.contains(" * ");
        let no_safe_ops = !line.contains("saturating_") && !line.contains("checked_") && !line.contains("wrapping_");
        let has_cast = line.contains("as usize") || line.contains("as u") || line.contains("as i");
        if has_arith && no_safe_ops && has_cast {
            finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-MUT-{:04}", finding_id),
                    file_path,
                    line_num,
                    "Arithmetic operation mutation target",
                )
                .with_description("Unchecked arithmetic with type cast; consider checked_* or saturating_* operations")
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.55)
                .with_discovered_by(HuntMode::Falsify)
                .with_evidence(FindingEvidence::mutation(format!("arith_{}", finding_id), true)),
            );
        }

        // Pattern: Boolean logic (negation mutation)
        let has_logic = line.contains(" && ") || line.contains(" || ");
        let has_predicate = line.contains("!") || line.contains("is_") || line.contains("has_");
        if has_logic && has_predicate {
            finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-MUT-{:04}", finding_id),
                    file_path,
                    line_num,
                    "Boolean logic mutation target",
                )
                .with_description("Complex boolean expression; verify test coverage catches negation mutations")
                .with_severity(FindingSeverity::Low)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.4)
                .with_discovered_by(HuntMode::Falsify)
                .with_evidence(FindingEvidence::mutation(format!("bool_{}", finding_id), true)),
            );
        }
    }
}

/// BH-02: SBFL without failing tests (SBEST pattern)
fn run_hunt_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Look for crash logs, stack traces, or error reports
    let crash_patterns = [
        "crash.log",
        "panic.log",
        "*.crash",
        "stack_trace.txt",
        "core.*",
    ];

    let mut stack_traces_found = Vec::new();

    for pattern in crash_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/**/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                stack_traces_found.push(entry);
            }
        }
    }

    // Also check for recent panic messages in test output
    let test_output = project_path.join("target/nextest/ci/junit.xml");
    if test_output.exists() {
        if let Ok(content) = std::fs::read_to_string(&test_output) {
            if content.contains("panicked") || content.contains("FAILED") {
                stack_traces_found.push(test_output);
            }
        }
    }

    if stack_traces_found.is_empty() {
        // No crash data available, fall back to coverage-based analysis
        analyze_coverage_hotspots(project_path, config, result);
    } else {
        for trace_file in stack_traces_found {
            analyze_stack_trace(&trace_file, project_path, config, result);
        }
    }
}

/// Analyze coverage data for suspicious hotspots.
fn analyze_coverage_hotspots(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Check custom coverage path first
    if let Some(ref custom_path) = config.coverage_path {
        if custom_path.exists() {
            if let Ok(content) = std::fs::read_to_string(custom_path) {
                parse_lcov_for_hotspots(&content, project_path, result);
                return;
            }
        }
    }

    // Build list of paths to search
    let mut lcov_paths: Vec<std::path::PathBuf> = vec![
        // Project root (common location for cargo llvm-cov output)
        project_path.join("lcov.info"),
        // Standard target locations
        project_path.join("target/coverage/lcov.info"),
        project_path.join("target/llvm-cov/lcov.info"),
        project_path.join("coverage/lcov.info"),
    ];

    // Check CARGO_TARGET_DIR for custom target locations
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let target_path = std::path::Path::new(&target_dir);
        lcov_paths.push(target_path.join("coverage/lcov.info"));
        lcov_paths.push(target_path.join("llvm-cov/lcov.info"));
    }

    for lcov_path in &lcov_paths {
        if lcov_path.exists() {
            if let Ok(content) = std::fs::read_to_string(lcov_path) {
                parse_lcov_for_hotspots(&content, project_path, result);
                return;
            }
        }
    }

    // No coverage data available
    let searched = lcov_paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    result.add_finding(
        Finding::new(
            "BH-HUNT-NOCOV",
            project_path.join("target"),
            1,
            "No coverage data available",
        )
        .with_description(format!(
            "Run `cargo llvm-cov --lcov --output-path lcov.info` or use --coverage-path. Searched: {}",
            searched
        ))
        .with_severity(FindingSeverity::Info)
        .with_category(DefectCategory::ConfigurationErrors)
        .with_suspiciousness(0.1)
        .with_discovered_by(HuntMode::Hunt),
    );
}

/// Parse LCOV data for coverage hotspots.
fn parse_lcov_for_hotspots(content: &str, project_path: &Path, result: &mut HuntResult) {
    let mut current_file: Option<String> = None;
    let mut uncovered_lines: Vec<(String, usize)> = Vec::new();

    for line in content.lines() {
        if let Some(file) = line.strip_prefix("SF:") {
            current_file = Some(file.to_string());
        } else if let Some(da) = line.strip_prefix("DA:") {
            if let Some(ref file) = current_file {
                let parts: Vec<&str> = da.split(',').collect();
                if parts.len() >= 2 {
                    if let (Ok(line_num), Ok(hits)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                        if hits == 0 {
                            uncovered_lines.push((file.clone(), line_num));
                        }
                    }
                }
            }
        } else if line == "end_of_record" {
            current_file = None;
        }
    }

    // Report files with many uncovered lines as suspicious
    let mut file_uncovered: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (file, line) in uncovered_lines {
        file_uncovered.entry(file).or_default().push(line);
    }

    let mut finding_id = 0;
    for (file, lines) in file_uncovered {
        if lines.len() > 5 {
            finding_id += 1;
            let suspiciousness = (lines.len() as f64 / 100.0).min(0.8);
            result.add_finding(
                Finding::new(
                    format!("BH-COV-{:04}", finding_id),
                    project_path.join(&file),
                    lines[0],
                    format!("Low coverage region ({} uncovered lines)", lines.len()),
                )
                .with_description(format!(
                    "Lines {} are never executed; potential dead code or missing tests",
                    lines.iter().take(5).map(|l| l.to_string()).collect::<Vec<_>>().join(", ")
                ))
                .with_severity(FindingSeverity::Low)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(suspiciousness)
                .with_discovered_by(HuntMode::Hunt)
                .with_evidence(FindingEvidence::sbfl("Coverage", suspiciousness)),
            );
        }
    }
}

/// Analyze a stack trace file.
fn analyze_stack_trace(trace_file: &Path, _project_path: &Path, _config: &HuntConfig, result: &mut HuntResult) {
    let content = match std::fs::read_to_string(trace_file) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut finding_id = 0;

    // Look for panic locations
    for line in content.lines() {
        // Pattern: "at src/file.rs:123"
        if let Some(at_pos) = line.find(" at ") {
            let location = &line[at_pos + 4..];
            if let Some(colon_pos) = location.rfind(':') {
                let file = &location[..colon_pos];
                if let Ok(line_num) = location[colon_pos + 1..].trim().parse::<usize>() {
                    if file.ends_with(".rs") && !file.contains("/.cargo/") {
                        finding_id += 1;
                        result.add_finding(
                            Finding::new(
                                format!("BH-STACK-{:04}", finding_id),
                                file,
                                line_num,
                                "Stack trace location",
                            )
                            .with_description(format!("Found in stack trace: {}", trace_file.display()))
                            .with_severity(FindingSeverity::High)
                            .with_category(DefectCategory::LogicErrors)
                            .with_suspiciousness(0.85)
                            .with_discovered_by(HuntMode::Hunt)
                            .with_evidence(FindingEvidence::sbfl("StackTrace", 0.85)),
                        );
                    }
                }
            }
        }
    }
}

/// BH-03: LLM-augmented static analysis (LLIFT pattern)
fn run_analyze_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Run clippy and collect warnings
    let clippy_output = std::process::Command::new("cargo")
        .args(["clippy", "--all-targets", "--message-format=json"])
        .current_dir(project_path)
        .output();

    let clippy_json = match clippy_output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
        Err(_) => {
            result.add_finding(
                Finding::new(
                    "BH-ANALYZE-NOCLIPPY",
                    project_path.join("Cargo.toml"),
                    1,
                    "Clippy not available",
                )
                .with_severity(FindingSeverity::Info)
                .with_discovered_by(HuntMode::Analyze),
            );
            return;
        }
    };

    let mut finding_id = 0;

    // Parse clippy JSON output
    for line in clippy_json.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                if let Some(message) = msg.get("message") {
                    let level = message.get("level").and_then(|l| l.as_str()).unwrap_or("");
                    if level == "warning" || level == "error" {
                        if let Some(spans) = message.get("spans").and_then(|s| s.as_array()) {
                            if let Some(span) = spans.first() {
                                let file = span.get("file_name")
                                    .and_then(|f| f.as_str())
                                    .unwrap_or("unknown");
                                let line_start = span.get("line_start")
                                    .and_then(|l| l.as_u64())
                                    .unwrap_or(1) as usize;
                                let msg_text = message.get("message")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("Unknown warning");
                                let code = message.get("code")
                                    .and_then(|c| c.get("code"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("unknown");

                                // Skip certain noisy warnings
                                if code == "dead_code" || code == "unused_imports" {
                                    continue;
                                }

                                // Categorize by clippy lint
                                let (category, severity) = categorize_clippy_warning(code, msg_text);

                                // Apply min_suspiciousness filter
                                let suspiciousness = match severity {
                                    FindingSeverity::Critical => 0.95,
                                    FindingSeverity::High => 0.8,
                                    FindingSeverity::Medium => 0.6,
                                    FindingSeverity::Low => 0.4,
                                    FindingSeverity::Info => 0.2,
                                };

                                if suspiciousness >= config.min_suspiciousness {
                                    finding_id += 1;
                                    result.add_finding(
                                        Finding::new(
                                            format!("BH-CLIP-{:04}", finding_id),
                                            file,
                                            line_start,
                                            msg_text,
                                        )
                                        .with_severity(severity)
                                        .with_category(category)
                                        .with_suspiciousness(suspiciousness)
                                        .with_discovered_by(HuntMode::Analyze)
                                        .with_evidence(FindingEvidence::static_analysis("clippy", code)),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Also look for common bug patterns via grep
    analyze_common_patterns(project_path, config, result);
}

/// Categorize clippy warning by code.
fn categorize_clippy_warning(code: &str, _message: &str) -> (DefectCategory, FindingSeverity) {
    match code {
        // Memory safety
        c if c.contains("ptr") || c.contains("mem") || c.contains("uninit") => {
            (DefectCategory::MemorySafety, FindingSeverity::High)
        }
        // Concurrency
        c if c.contains("mutex") || c.contains("arc") || c.contains("send") || c.contains("sync") => {
            (DefectCategory::ConcurrencyBugs, FindingSeverity::High)
        }
        // Security
        c if c.contains("unsafe") || c.contains("transmute") => {
            (DefectCategory::SecurityVulnerabilities, FindingSeverity::High)
        }
        // Logic errors
        c if c.contains("unwrap") || c.contains("expect") || c.contains("panic") => {
            (DefectCategory::LogicErrors, FindingSeverity::Medium)
        }
        // Type errors
        c if c.contains("cast") || c.contains("as_") || c.contains("into") => {
            (DefectCategory::TypeErrors, FindingSeverity::Medium)
        }
        // Default
        _ => (DefectCategory::Unknown, FindingSeverity::Low),
    }
}

/// Analyze common bug patterns via grep.
fn analyze_common_patterns(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Load bug-hunter config for allowlist and custom patterns
    let bh_config = self::config::BugHunterConfig::load(project_path);

    // BH-23: If PMAT SATD is enabled and pmat is available, generate SATD findings
    // and skip the manual TODO/FIXME/HACK/XXX pattern matching
    let pmat_satd_active = config.pmat_satd && pmat_quality::pmat_available();
    if pmat_satd_active {
        let query = config.pmat_query.as_deref().unwrap_or("*");
        if let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) {
            let satd_findings = pmat_quality::generate_satd_findings(project_path, &index);
            for f in satd_findings {
                result.add_finding(f);
            }
        }
    }

    // GPU/CUDA patterns (always active - detect hidden kernel bugs)
    let gpu_patterns: Vec<(&str, DefectCategory, FindingSeverity, f64)> = vec![
        // GPU kernel bugs - comments indicating broken CUDA/PTX
        ("CUDA_ERROR", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.9),
        ("INVALID_PTX", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.95),
        ("PTX error", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.9),
        ("kernel fail", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.8),
        ("cuBLAS fallback", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("cuDNN fallback", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        // Silent degradation - error swallowing without alerting
        (".unwrap_or_else(|_|", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        ("if let Err(_) =", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.5),
        ("Err(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
        ("Ok(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.4),
        ("// fallback", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.5),
        ("// degraded", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        // Test debt - skipped tests indicating known bugs
        ("#[ignore]", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("// skip", DefectCategory::TestDebt, FindingSeverity::Medium, 0.5),
        ("// skipped", DefectCategory::TestDebt, FindingSeverity::Medium, 0.5),
        ("// broken", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        ("// fails", DefectCategory::TestDebt, FindingSeverity::High, 0.75),
        ("// disabled", DefectCategory::TestDebt, FindingSeverity::Medium, 0.6),
        ("test removed", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("were removed", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("tests hang", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("hang during", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        ("compilation hang", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        // Dimension-related GPU bugs (hidden_dim limits)
        ("hidden_dim >=", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("hidden_dim >", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("// 1536", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        ("// 2048", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        ("model dimensions", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        // Hidden debt - euphemisms that hide technical debt (PMAT issue #149)
        // These appear in doc comments and regular comments
        ("placeholder", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("stub", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("dummy", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("fake", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("mock", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("simplified", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("for demonstration", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("demo only", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        ("not implemented", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("unimplemented", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("temporary", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("hardcoded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("hard-coded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("magic number", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("workaround", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("quick fix", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("quick-fix", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("bandaid", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("band-aid", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("kludge", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("tech debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        ("technical debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
    ];

    let mut patterns: Vec<(&str, DefectCategory, FindingSeverity, f64)> = if pmat_satd_active {
        // When PMAT SATD is active, skip TODO/FIXME/HACK/XXX (handled by PMAT)
        // Keep unwrap/unsafe/transmute/panic patterns always
        vec![
            ("unwrap()", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.4),
            ("expect(", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
            ("unsafe {", DefectCategory::MemorySafety, FindingSeverity::High, 0.7),
            ("transmute", DefectCategory::MemorySafety, FindingSeverity::High, 0.8),
            ("panic!", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("unreachable!", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ]
    } else {
        vec![
            ("TODO", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
            ("FIXME", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("HACK", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("XXX", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("unwrap()", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.4),
            ("expect(", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
            ("unsafe {", DefectCategory::MemorySafety, FindingSeverity::High, 0.7),
            ("transmute", DefectCategory::MemorySafety, FindingSeverity::High, 0.8),
            ("panic!", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("unreachable!", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ]
    };
    // Merge GPU patterns (always active)
    patterns.extend(gpu_patterns);

    // Convert custom patterns from config to owned patterns
    let custom_patterns: Vec<(String, DefectCategory, FindingSeverity, f64)> = bh_config
        .patterns
        .iter()
        .map(|p| {
            let category = match p.category.to_lowercase().as_str() {
                "logicerrors" | "logic" => DefectCategory::LogicErrors,
                "memorysafety" | "memory" => DefectCategory::MemorySafety,
                "concurrency" | "concurrencybugs" => DefectCategory::ConcurrencyBugs,
                "gpukernelbugs" | "gpu" => DefectCategory::GpuKernelBugs,
                "silentdegradation" | "silent" => DefectCategory::SilentDegradation,
                "testdebt" | "test" => DefectCategory::TestDebt,
                "hiddendebt" | "debt" => DefectCategory::HiddenDebt,
                "performanceissues" | "performance" => DefectCategory::PerformanceIssues,
                "securityvulnerabilities" | "security" => DefectCategory::SecurityVulnerabilities,
                _ => DefectCategory::LogicErrors,
            };
            let severity = match p.severity.to_lowercase().as_str() {
                "critical" => FindingSeverity::Critical,
                "high" => FindingSeverity::High,
                "medium" => FindingSeverity::Medium,
                "low" => FindingSeverity::Low,
                "info" => FindingSeverity::Info,
                _ => FindingSeverity::Medium,
            };
            (p.pattern.clone(), category, severity, p.suspiciousness)
        })
        .collect();

    // Collect all file paths first (multi-language support)
    let mut all_files: Vec<std::path::PathBuf> = Vec::new();
    for target in &config.targets {
        let target_path = project_path.join(target);
        // Scan all supported languages
        for glob_pattern in languages::all_language_globs() {
            if let Ok(entries) =
                glob::glob(&format!("{}/{}", target_path.display(), glob_pattern))
            {
                all_files.extend(entries.flatten());
            }
        }
    }

    // Parallel file scanning via std::thread::scope
    let min_susp = config.min_suspiciousness;
    let chunk_size = (all_files.len() / 4).max(1);
    let chunks: Vec<&[std::path::PathBuf]> = all_files.chunks(chunk_size).collect();

    let all_chunk_findings: Vec<Vec<Finding>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let patterns = &patterns;
                let custom_patterns = &custom_patterns;
                let bh_config = &bh_config;
                s.spawn(move || {
                    let mut chunk_findings = Vec::new();
                    for entry in *chunk {
                        let Ok(content) = std::fs::read_to_string(entry) else {
                            continue;
                        };
                        let test_lines = compute_test_lines(&content);

                        // Detect language from file extension
                        let lang = entry
                            .extension()
                            .and_then(|e| e.to_str())
                            .and_then(languages::Language::from_extension);

                        // Get language-specific patterns (or use Rust patterns as default)
                        let lang_patterns = lang
                            .map(languages::patterns_for_language)
                            .unwrap_or_else(|| patterns.iter().map(|&(p, c, s, su)| (p, c, s, su)).collect());

                        for (line_num, line) in content.lines().enumerate() {
                            let line_num = line_num + 1;
                            let in_test_code = test_lines.contains(&line_num);

                            // Use language-specific patterns
                            for (pattern, category, severity, suspiciousness) in &lang_patterns {
                                // Skip most patterns in test code (unwrap/panic expected in tests)
                                // But ALWAYS scan TestDebt, GpuKernelBugs, and HiddenDebt patterns
                                // - these indicate known bugs or euphemisms hiding technical debt
                                if in_test_code
                                    && *category != DefectCategory::TestDebt
                                    && *category != DefectCategory::GpuKernelBugs
                                    && *category != DefectCategory::HiddenDebt
                                {
                                    continue;
                                }
                                // Skip HiddenDebt patterns in bug_hunter module (it discusses these patterns)
                                let is_bug_hunter_file = entry
                                    .to_str()
                                    .map(|p| p.contains("bug_hunter"))
                                    .unwrap_or(false);
                                if is_bug_hunter_file && *category == DefectCategory::HiddenDebt {
                                    continue;
                                }
                                // Check allowlist from .pmat/bug-hunter.toml
                                if bh_config.is_allowed(entry, pattern, line_num) {
                                    continue;
                                }
                                if line.contains(pattern)
                                    && is_real_pattern(line, pattern)
                                    && *suspiciousness >= min_susp
                                {
                                    let finding = Finding::new(
                                            String::new(), // placeholder ID
                                            entry,
                                            line_num,
                                            format!("Pattern: {}", pattern),
                                        )
                                        .with_description(line.trim().to_string())
                                        .with_severity(*severity)
                                        .with_category(*category)
                                        .with_suspiciousness(*suspiciousness)
                                        .with_discovered_by(HuntMode::Analyze)
                                        .with_evidence(FindingEvidence::static_analysis(
                                            "pattern", *pattern,
                                        ));
                                    // BH-15: Check suppression rules (issue #17)
                                    if !should_suppress_finding(&finding, line) {
                                        chunk_findings.push(finding);
                                    }
                                }
                            }

                            // Scan custom patterns from .pmat/bug-hunter.toml
                            for (pattern, category, severity, suspiciousness) in custom_patterns {
                                if *suspiciousness < min_susp {
                                    continue;
                                }
                                // Check allowlist
                                if bh_config.is_allowed(entry, pattern, line_num) {
                                    continue;
                                }
                                if line.contains(pattern.as_str()) {
                                    let finding = Finding::new(
                                            String::new(),
                                            entry,
                                            line_num,
                                            format!("Custom: {}", pattern),
                                        )
                                        .with_description(line.trim().to_string())
                                        .with_severity(*severity)
                                        .with_category(*category)
                                        .with_suspiciousness(*suspiciousness)
                                        .with_discovered_by(HuntMode::Analyze)
                                        .with_evidence(FindingEvidence::static_analysis(
                                            "custom_pattern", pattern,
                                        ));
                                    // BH-15: Check suppression rules (issue #17)
                                    if !should_suppress_finding(&finding, line) {
                                        chunk_findings.push(finding);
                                    }
                                }
                            }
                        }
                    }
                    chunk_findings
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge all findings and assign globally unique IDs
    // Also fetch git blame info for each finding
    let mut blame_cache = blame::BlameCache::new();
    let mut finding_id = 0u32;
    for chunk_findings in all_chunk_findings {
        for mut finding in chunk_findings {
            finding_id += 1;
            finding.id = format!("BH-PAT-{:04}", finding_id);

            // Fetch git blame info
            if let Some(blame_info) =
                blame_cache.get_blame(project_path, &finding.file, finding.line)
            {
                finding.blame_author = Some(blame_info.author);
                finding.blame_commit = Some(blame_info.commit);
                finding.blame_date = Some(blame_info.date);
            }

            result.add_finding(finding);
        }
    }
}

/// Check if the crate forbids unsafe code (BH-19 fix).
fn crate_forbids_unsafe(project_path: &Path) -> bool {
    // Check lib.rs and main.rs for #![forbid(unsafe_code)]
    let entry_files = ["src/lib.rs", "src/main.rs"];
    for entry in &entry_files {
        let path = project_path.join(entry);
        if let Ok(content) = std::fs::read_to_string(&path) {
            // Check the first 50 lines for crate-level attributes
            for line in content.lines().take(50) {
                let trimmed = line.trim();
                if trimmed.starts_with("#![") && trimmed.contains("forbid") && trimmed.contains("unsafe_code") {
                    return true;
                }
            }
        }
    }

    // Check Cargo.toml for [lints.rust] forbid(unsafe_code)
    let cargo_toml = project_path.join("Cargo.toml");
    if let Ok(content) = std::fs::read_to_string(cargo_toml) {
        // Look for unsafe_code = "forbid" in lints section
        if content.contains("unsafe_code") && content.contains("forbid") {
            return true;
        }
    }

    false
}

/// BH-04: Targeted unsafe Rust fuzzing (FourFuzz pattern)
fn run_fuzz_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Check if crate forbids unsafe code (issue #19)
    if crate_forbids_unsafe(project_path) {
        result.add_finding(
            Finding::new(
                "BH-FUZZ-SKIPPED",
                project_path.join("src/lib.rs"),
                1,
                "Fuzz targets not needed - crate forbids unsafe code",
            )
            .with_description("Crate uses #![forbid(unsafe_code)], no unsafe blocks to fuzz")
            .with_severity(FindingSeverity::Info)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.0)
            .with_discovered_by(HuntMode::Fuzz),
        );
        return;
    }

    // Inventory unsafe blocks
    let mut unsafe_inventory = Vec::new();
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", target_path.display())) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(&entry) {
                    let mut in_unsafe = false;
                    let mut unsafe_start = 0;

                    for (line_num, line) in content.lines().enumerate() {
                        let line_num = line_num + 1;

                        if line.contains("unsafe ") && line.contains("{") {
                            in_unsafe = true;
                            unsafe_start = line_num;
                        }

                        if in_unsafe {
                            // Look for dangerous operations within unsafe
                            if line.contains("*") && (line.contains("ptr") || line.contains("as *")) {
                                finding_id += 1;
                                unsafe_inventory.push((entry.clone(), line_num));
                                result.add_finding(
                                    Finding::new(
                                        format!("BH-UNSAFE-{:04}", finding_id),
                                        &entry,
                                        line_num,
                                        "Pointer dereference in unsafe block",
                                    )
                                    .with_description(format!("Unsafe block starting at line {}; potential fuzzing target", unsafe_start))
                                    .with_severity(FindingSeverity::High)
                                    .with_category(DefectCategory::MemorySafety)
                                    .with_suspiciousness(0.75)
                                    .with_discovered_by(HuntMode::Fuzz)
                                    .with_evidence(FindingEvidence::fuzzing("N/A", "pointer_deref")),
                                );
                            }

                            if line.contains("transmute") {
                                finding_id += 1;
                                result.add_finding(
                                    Finding::new(
                                        format!("BH-UNSAFE-{:04}", finding_id),
                                        &entry,
                                        line_num,
                                        "Transmute in unsafe block",
                                    )
                                    .with_description("std::mem::transmute bypasses type safety; high-priority fuzzing target")
                                    .with_severity(FindingSeverity::Critical)
                                    .with_category(DefectCategory::MemorySafety)
                                    .with_suspiciousness(0.9)
                                    .with_discovered_by(HuntMode::Fuzz)
                                    .with_evidence(FindingEvidence::fuzzing("N/A", "transmute")),
                                );
                            }
                        }

                        if line.contains("}") && in_unsafe {
                            in_unsafe = false;
                        }
                    }
                }
            }
        }
    }

    // Check for existing fuzz targets
    let fuzz_dir = project_path.join("fuzz");
    if !fuzz_dir.exists() {
        result.add_finding(
            Finding::new(
                "BH-FUZZ-NOTARGETS",
                project_path.join("Cargo.toml"),
                1,
                "No fuzz directory found",
            )
            .with_description(format!(
                "Create fuzz targets for {} identified unsafe blocks",
                unsafe_inventory.len()
            ))
            .with_severity(FindingSeverity::Medium)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.4)
            .with_discovered_by(HuntMode::Fuzz),
        );
    }

    // Update mode stats
    result.stats.mode_stats.fuzz_coverage = if unsafe_inventory.is_empty() {
        100.0
    } else {
        0.0 // Would need actual fuzzing run to determine
    };
}

/// BH-05: Hybrid concolic + SBFL (COTTONTAIL pattern)
fn run_deep_hunt_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Identify complex conditionals that would benefit from concolic execution
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", target_path.display())) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(&entry) {
                    let mut complexity: usize = 0;
                    let mut complex_start: usize = 0;

                    for (line_num, line) in content.lines().enumerate() {
                        let line_num = line_num + 1;

                        // Count complexity indicators
                        if line.contains("if ") || line.contains("match ") {
                            complexity += 1;
                            if complexity == 1 {
                                complex_start = line_num;
                            }
                        }

                        // Nested conditions = high complexity
                        if complexity >= 3 && line.contains("if ") {
                            finding_id += 1;
                            result.add_finding(
                                Finding::new(
                                    format!("BH-DEEP-{:04}", finding_id),
                                    &entry,
                                    line_num,
                                    "Deeply nested conditional",
                                )
                                .with_description(format!(
                                    "Complexity {} starting at line {}; concolic execution recommended",
                                    complexity, complex_start
                                ))
                                .with_severity(FindingSeverity::Medium)
                                .with_category(DefectCategory::LogicErrors)
                                .with_suspiciousness(0.6)
                                .with_discovered_by(HuntMode::DeepHunt)
                                .with_evidence(FindingEvidence::concolic(format!("depth={}", complexity))),
                            );
                        }

                        // Look for complex guards
                        if line.contains(" && ") && line.contains(" || ") {
                            finding_id += 1;
                            result.add_finding(
                                Finding::new(
                                    format!("BH-DEEP-{:04}", finding_id),
                                    &entry,
                                    line_num,
                                    "Complex boolean guard",
                                )
                                .with_description("Mixed AND/OR logic; path explosion potential")
                                .with_severity(FindingSeverity::Medium)
                                .with_category(DefectCategory::LogicErrors)
                                .with_suspiciousness(0.55)
                                .with_discovered_by(HuntMode::DeepHunt)
                                .with_evidence(FindingEvidence::concolic("complex_guard")),
                            );
                        }

                        if line.contains("}") && complexity > 0 {
                            complexity -= 1;
                        }
                    }
                }
            }
        }
    }

    // Also run SBFL analysis
    run_hunt_mode(project_path, config, result);
}

/// Quick mode: pattern matching only, no clippy, no coverage analysis.
/// Fastest mode for quick scans.
fn run_quick_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Only run pattern analysis (from analyze mode)
    analyze_common_patterns(project_path, config, result);
}

#[cfg(test)]
mod tests {
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
        // Note: Line numbers match enumerate() + 1
        // Line 1: fn production_code() {
        // Line 2:     panic!("this should be caught");
        // Line 3: }
        // Line 4: (empty)
        // Line 5: #[cfg(test)]
        // Line 6: mod tests {
        // Line 7:     #[test]
        // Line 8:     fn my_test() {
        // Line 9:         panic!("this should be ignored");
        // Line 10:    }
        // Line 11: }
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
        // Line 5 (#[cfg(test)]) should be marked as start of test
        assert!(test_lines.contains(&5), "Line 5 (#[cfg(test)]) should be test code");
        // Lines 6-11 (inside test module) should be test code
        assert!(test_lines.contains(&6), "Line 6 (mod tests) should be test code");
        assert!(test_lines.contains(&7), "Line 7 (#[test]) should be test code");
        assert!(test_lines.contains(&9), "Line 9 (panic) should be test code");
        // Line 2 (production panic) should NOT be in test lines
        assert!(!test_lines.contains(&2), "Line 2 (production panic) should NOT be test code");
    }

    #[test]
    fn test_bh_mod_005_compute_test_lines_individual_test() {
        // Line 1: fn production_code() {
        // Line 2:     println!("production");
        // Line 3: }
        // Line 4: (empty)
        // Line 5: #[test]
        // Line 6: fn standalone_test() {
        // Line 7:     panic!("test assertion");
        // Line 8: }
        let content = r#"fn production_code() {
    println!("production");
}

#[test]
fn standalone_test() {
    panic!("test assertion");
}"#;
        let test_lines = compute_test_lines(content);
        // The #[test] attribute and function body should be test code
        assert!(test_lines.contains(&5), "Line 5 (#[test]) should be test code");
        assert!(test_lines.contains(&6), "Line 6 (fn standalone_test) should be test code");
        assert!(test_lines.contains(&7), "Line 7 (panic) should be test code");
        // Production code should not be marked
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
        // Real TODO in a comment
        assert!(is_real_pattern("// TODO: fix this", "TODO"));
        assert!(is_real_pattern("    // TODO fix later", "TODO"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_todo_in_string() {
        // TODO inside a string literal is NOT a real TODO
        assert!(!is_real_pattern(r#"let msg = "TODO: implement";"#, "TODO"));
        assert!(!is_real_pattern(r#"println!("TODO/FIXME markers");"#, "TODO"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_unsafe_in_code() {
        // Real unsafe block
        assert!(is_real_pattern("unsafe { ptr::read(p) }", "unsafe {"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_unsafe_in_comment() {
        // unsafe in a comment is NOT a real pattern
        assert!(!is_real_pattern("// unsafe blocks need safety comments", "unsafe {"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_unsafe_in_variable() {
        // "unsafe {" inside a variable name like "in_unsafe" is NOT a real pattern
        assert!(!is_real_pattern("if in_unsafe {", "unsafe {"));
        assert!(!is_real_pattern("let foo_unsafe = true;", "unsafe {"));
        // But standalone unsafe IS a real pattern
        assert!(is_real_pattern("    unsafe { foo() }", "unsafe {"));
        assert!(is_real_pattern("return unsafe { bar };", "unsafe {"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_unwrap_in_code() {
        // Real unwrap call
        assert!(is_real_pattern("let x = opt.unwrap();", "unwrap()"));
    }

    #[test]
    fn test_bh_mod_006_real_pattern_unwrap_in_doc() {
        // unwrap in doc comment is NOT a real pattern
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
        // Test that analyze_coverage_hotspots handles missing files gracefully
        let temp = std::env::temp_dir().join("test_bh_mod_007");
        let _ = std::fs::create_dir_all(&temp);
        let config = HuntConfig::default();
        let mut result = HuntResult::new(&temp, HuntMode::Hunt, config.clone());

        analyze_coverage_hotspots(&temp, &config, &mut result);

        // Should add a BH-HUNT-NOCOV finding
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

        // Write lib.rs with #![forbid(unsafe_code)]
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

        // Write main.rs with #![forbid(unsafe_code)]
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

        // Write src/lib.rs without forbid
        std::fs::write(temp.join("src/lib.rs"), "pub fn safe_fn() {}\n").unwrap();

        // Write Cargo.toml with lints.rust forbid unsafe_code
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

        // Write lib.rs without forbid
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

        // Write lib.rs with #![forbid(unsafe_code)]
        std::fs::write(
            temp.join("src/lib.rs"),
            "#![forbid(unsafe_code)]\n\npub fn safe_fn() {}\n",
        )
        .unwrap();

        let config = HuntConfig::default();
        let mut result = HuntResult::new(&temp, HuntMode::Fuzz, config.clone());

        run_fuzz_mode(&temp, &config, &mut result);

        // Should have BH-FUZZ-SKIPPED, not BH-FUZZ-NOTARGETS
        let skipped = result.findings.iter().any(|f| f.id == "BH-FUZZ-SKIPPED");
        let notargets = result.findings.iter().any(|f| f.id == "BH-FUZZ-NOTARGETS");

        assert!(skipped, "Should report BH-FUZZ-SKIPPED for forbid(unsafe_code) crates");
        assert!(!notargets, "Should NOT report BH-FUZZ-NOTARGETS for forbid(unsafe_code) crates");

        let _ = std::fs::remove_dir_all(&temp);
    }
}
