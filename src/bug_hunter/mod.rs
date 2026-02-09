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
        apply_spec_quality_gate(&mut parsed_spec, project_path, &mut result, query);
    }

    // Update spec claims with findings
    let findings_by_claim: Vec<(String, Vec<Finding>)> = mapping.into_iter().collect();
    if let Ok(updated_content) = parsed_spec.update_with_findings(&findings_by_claim) {
        parsed_spec.original_content = updated_content;
    }

    result.duration_ms = start.elapsed().as_millis() as u64;

    Ok((result, parsed_spec))
}

/// Apply quality gate checks to spec claims using PMAT quality index (BH-25).
fn apply_spec_quality_gate(
    parsed_spec: &mut ParsedSpec,
    project_path: &Path,
    result: &mut HuntResult,
    query: &str,
) {
    let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) else {
        return;
    };
    for claim in &mut parsed_spec.claims {
        for imp in &claim.implementations {
            let Some(pmat) = pmat_quality::lookup_quality(&index, &imp.file, imp.line) else {
                continue;
            };
            let is_low_quality =
                pmat.tdg_grade == "D" || pmat.tdg_grade == "F" || pmat.complexity > 20;
            if !is_low_quality {
                continue;
            }
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

/// Detected mutation target with metadata.
struct MutationMatch {
    title: &'static str,
    description: &'static str,
    severity: FindingSeverity,
    suspiciousness: f64,
    prefix: &'static str,
}

/// Detect mutation targets in a single line of code.
fn detect_mutation_targets(line: &str) -> Vec<MutationMatch> {
    let mut matches = Vec::new();

    let has_comparison =
        line.contains("< ") || line.contains("> ") || line.contains("<= ") || line.contains(">= ");
    let has_len = line.contains("len()") || line.contains("size()") || line.contains(".len");
    if has_comparison && has_len {
        matches.push(MutationMatch {
            title: "Boundary condition mutation target",
            description: "Off-by-one errors are common; this comparison should be mutation-tested",
            severity: FindingSeverity::Medium,
            suspiciousness: 0.6,
            prefix: "boundary",
        });
    }

    let has_arith = line.contains(" + ") || line.contains(" - ") || line.contains(" * ");
    let no_safe =
        !line.contains("saturating_") && !line.contains("checked_") && !line.contains("wrapping_");
    let has_cast =
        line.contains("as usize") || line.contains("as u") || line.contains("as i");
    if has_arith && no_safe && has_cast {
        matches.push(MutationMatch {
            title: "Arithmetic operation mutation target",
            description:
                "Unchecked arithmetic with type cast; consider checked_* or saturating_* operations",
            severity: FindingSeverity::Medium,
            suspiciousness: 0.55,
            prefix: "arith",
        });
    }

    let has_logic = line.contains(" && ") || line.contains(" || ");
    let has_predicate = line.contains("!") || line.contains("is_") || line.contains("has_");
    if has_logic && has_predicate {
        matches.push(MutationMatch {
            title: "Boolean logic mutation target",
            description:
                "Complex boolean expression; verify test coverage catches negation mutations",
            severity: FindingSeverity::Low,
            suspiciousness: 0.4,
            prefix: "bool",
        });
    }

    matches
}

/// Analyze a file for mutation testing targets.
fn analyze_file_for_mutations(file_path: &Path, _config: &HuntConfig, result: &mut HuntResult) {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return;
    };

    let mut finding_id = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        for m in detect_mutation_targets(line) {
            finding_id += 1;
            result.add_finding(
                Finding::new(format!("BH-MUT-{:04}", finding_id), file_path, line_num, m.title)
                    .with_description(m.description)
                    .with_severity(m.severity)
                    .with_category(DefectCategory::LogicErrors)
                    .with_suspiciousness(m.suspiciousness)
                    .with_discovered_by(HuntMode::Falsify)
                    .with_evidence(FindingEvidence::mutation(
                        format!("{}_{}", m.prefix, finding_id),
                        true,
                    )),
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

/// Parse a single DA line from LCOV data, recording uncovered lines.
fn parse_lcov_da_line(
    da: &str,
    file: &str,
    file_uncovered: &mut std::collections::HashMap<String, Vec<usize>>,
) {
    let Some((line_str, hits_str)) = da.split_once(',') else {
        return;
    };
    let Ok(line_num) = line_str.parse::<usize>() else {
        return;
    };
    let Ok(hits) = hits_str.parse::<usize>() else {
        return;
    };
    if hits == 0 {
        file_uncovered
            .entry(file.to_string())
            .or_default()
            .push(line_num);
    }
}

/// Report files with many uncovered lines as suspicious findings.
fn report_uncovered_hotspots(
    file_uncovered: std::collections::HashMap<String, Vec<usize>>,
    project_path: &Path,
    result: &mut HuntResult,
) {
    let mut finding_id = 0;
    for (file, lines) in file_uncovered {
        if lines.len() <= 5 {
            continue;
        }
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
                lines
                    .iter()
                    .take(5)
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .with_severity(FindingSeverity::Low)
            .with_category(DefectCategory::LogicErrors)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Hunt)
            .with_evidence(FindingEvidence::sbfl("Coverage", suspiciousness)),
        );
    }
}

/// Parse LCOV data for coverage hotspots.
fn parse_lcov_for_hotspots(content: &str, project_path: &Path, result: &mut HuntResult) {
    let mut current_file: Option<String> = None;
    let mut file_uncovered: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for line in content.lines() {
        if let Some(file) = line.strip_prefix("SF:") {
            current_file = Some(file.to_string());
        } else if let Some(da) = line.strip_prefix("DA:") {
            if let Some(ref file) = current_file {
                parse_lcov_da_line(da, file, &mut file_uncovered);
            }
        } else if line == "end_of_record" {
            current_file = None;
        }
    }

    report_uncovered_hotspots(file_uncovered, project_path, result);
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

/// Extract a finding from a single clippy JSON message, if applicable.
fn extract_clippy_finding(
    msg: &serde_json::Value,
    config: &HuntConfig,
    finding_id: &mut usize,
) -> Option<Finding> {
    if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
        return None;
    }
    let message = msg.get("message")?;
    let level = message.get("level").and_then(|l| l.as_str()).unwrap_or("");
    if level != "warning" && level != "error" {
        return None;
    }
    let spans = message.get("spans").and_then(|s| s.as_array())?;
    let span = spans.first()?;
    let file = span
        .get("file_name")
        .and_then(|f| f.as_str())
        .unwrap_or("unknown");
    let line_start = span
        .get("line_start")
        .and_then(|l| l.as_u64())
        .unwrap_or(1) as usize;
    let msg_text = message
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("Unknown warning");
    let code = message
        .get("code")
        .and_then(|c| c.get("code"))
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");

    if code == "dead_code" || code == "unused_imports" {
        return None;
    }

    let (category, severity) = categorize_clippy_warning(code, msg_text);
    let suspiciousness = match severity {
        FindingSeverity::Critical => 0.95,
        FindingSeverity::High => 0.8,
        FindingSeverity::Medium => 0.6,
        FindingSeverity::Low => 0.4,
        FindingSeverity::Info => 0.2,
    };

    if suspiciousness < config.min_suspiciousness {
        return None;
    }

    *finding_id += 1;
    Some(
        Finding::new(format!("BH-CLIP-{:04}", finding_id), file, line_start, msg_text)
            .with_severity(severity)
            .with_category(category)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::static_analysis("clippy", code)),
    )
}

/// BH-03: LLM-augmented static analysis (LLIFT pattern)
fn run_analyze_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
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
    for line in clippy_json.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(finding) = extract_clippy_finding(&msg, config, &mut finding_id) {
                result.add_finding(finding);
            }
        }
    }

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
/// Parse defect category from string (for custom config patterns).
fn parse_defect_category(s: &str) -> DefectCategory {
    match s.to_lowercase().as_str() {
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
    }
}

/// Parse finding severity from string (for custom config patterns).
fn parse_finding_severity(s: &str) -> FindingSeverity {
    match s.to_lowercase().as_str() {
        "critical" => FindingSeverity::Critical,
        "high" => FindingSeverity::High,
        "medium" => FindingSeverity::Medium,
        "low" => FindingSeverity::Low,
        "info" => FindingSeverity::Info,
        _ => FindingSeverity::Medium,
    }
}

/// Context for pattern matching on a single source line.
struct PatternMatchContext<'a> {
    line: &'a str,
    line_num: usize,
    entry: &'a Path,
    in_test_code: bool,
    is_bug_hunter_file: bool,
    bh_config: &'a self::config::BugHunterConfig,
    min_susp: f64,
}

/// Check a single line against a language pattern, returning a finding if matched.
fn match_lang_pattern(
    ctx: &PatternMatchContext<'_>,
    pattern: &str,
    category: DefectCategory,
    severity: FindingSeverity,
    suspiciousness: f64,
) -> Option<Finding> {
    if ctx.in_test_code
        && category != DefectCategory::TestDebt
        && category != DefectCategory::GpuKernelBugs
        && category != DefectCategory::HiddenDebt
    {
        return None;
    }
    if ctx.is_bug_hunter_file && category == DefectCategory::HiddenDebt {
        return None;
    }
    if ctx.bh_config.is_allowed(ctx.entry, pattern, ctx.line_num) {
        return None;
    }
    if !ctx.line.contains(pattern)
        || !is_real_pattern(ctx.line, pattern)
        || suspiciousness < ctx.min_susp
    {
        return None;
    }
    let finding = Finding::new(
        String::new(),
        ctx.entry,
        ctx.line_num,
        format!("Pattern: {}", pattern),
    )
    .with_description(ctx.line.trim().to_string())
    .with_severity(severity)
    .with_category(category)
    .with_suspiciousness(suspiciousness)
    .with_discovered_by(HuntMode::Analyze)
    .with_evidence(FindingEvidence::static_analysis("pattern", pattern));
    if should_suppress_finding(&finding, ctx.line) {
        None
    } else {
        Some(finding)
    }
}

/// Check a single line against a custom pattern, returning a finding if matched.
fn match_custom_pattern(
    ctx: &PatternMatchContext<'_>,
    pattern: &str,
    category: DefectCategory,
    severity: FindingSeverity,
    suspiciousness: f64,
) -> Option<Finding> {
    if suspiciousness < ctx.min_susp || ctx.bh_config.is_allowed(ctx.entry, pattern, ctx.line_num) {
        return None;
    }
    if !ctx.line.contains(pattern) {
        return None;
    }
    let finding = Finding::new(
        String::new(),
        ctx.entry,
        ctx.line_num,
        format!("Custom: {}", pattern),
    )
    .with_description(ctx.line.trim().to_string())
    .with_severity(severity)
    .with_category(category)
    .with_suspiciousness(suspiciousness)
    .with_discovered_by(HuntMode::Analyze)
    .with_evidence(FindingEvidence::static_analysis("custom_pattern", pattern));
    if should_suppress_finding(&finding, ctx.line) {
        None
    } else {
        Some(finding)
    }
}

/// Scan a single file for pattern matches.
fn scan_file_for_patterns(
    entry: &std::path::Path,
    patterns: &[(&str, DefectCategory, FindingSeverity, f64)],
    custom_patterns: &[(String, DefectCategory, FindingSeverity, f64)],
    bh_config: &self::config::BugHunterConfig,
    min_susp: f64,
    findings: &mut Vec<Finding>,
) {
    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let test_lines = compute_test_lines(&content);
    let lang = entry
        .extension()
        .and_then(|e| e.to_str())
        .and_then(languages::Language::from_extension);
    let lang_patterns = lang
        .map(languages::patterns_for_language)
        .unwrap_or_else(|| patterns.iter().map(|&(p, c, s, su)| (p, c, s, su)).collect());
    let is_bug_hunter_file = entry
        .to_str()
        .map(|p| p.contains("bug_hunter"))
        .unwrap_or(false);

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        let ctx = PatternMatchContext {
            line,
            line_num,
            entry,
            in_test_code: test_lines.contains(&line_num),
            is_bug_hunter_file,
            bh_config,
            min_susp,
        };

        for &(pattern, category, severity, suspiciousness) in &lang_patterns {
            if let Some(f) = match_lang_pattern(&ctx, pattern, category, severity, suspiciousness) {
                findings.push(f);
            }
        }

        for (pattern, category, severity, suspiciousness) in custom_patterns {
            if let Some(f) =
                match_custom_pattern(&ctx, pattern.as_str(), *category, *severity, *suspiciousness)
            {
                findings.push(f);
            }
        }
    }
}

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
            let category = parse_defect_category(&p.category);
            let severity = parse_finding_severity(&p.severity);
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
                        scan_file_for_patterns(
                            entry, patterns, custom_patterns, bh_config, min_susp,
                            &mut chunk_findings,
                        );
                    }
                    chunk_findings
                })
            })
            .collect();

        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .collect()
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

/// Check if a source file contains #![forbid(unsafe_code)] in its first 50 lines.
fn source_forbids_unsafe(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    content.lines().take(50).any(|line| {
        let t = line.trim();
        t.starts_with("#![") && t.contains("forbid") && t.contains("unsafe_code")
    })
}

/// Check if the crate forbids unsafe code (BH-19 fix).
fn crate_forbids_unsafe(project_path: &Path) -> bool {
    for entry in ["src/lib.rs", "src/main.rs"] {
        if source_forbids_unsafe(&project_path.join(entry)) {
            return true;
        }
    }
    if let Ok(content) = std::fs::read_to_string(project_path.join("Cargo.toml")) {
        if content.contains("unsafe_code") && content.contains("forbid") {
            return true;
        }
    }
    false
}

/// Scan a single file for unsafe blocks and dangerous operations within them.
fn scan_file_for_unsafe_blocks(
    entry: &Path,
    finding_id: &mut usize,
    unsafe_inventory: &mut Vec<(std::path::PathBuf, usize)>,
    result: &mut HuntResult,
) {
    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let mut in_unsafe = false;
    let mut unsafe_start = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;

        if line.contains("unsafe ") && line.contains('{') {
            in_unsafe = true;
            unsafe_start = line_num;
        }

        if in_unsafe {
            if line.contains('*') && (line.contains("ptr") || line.contains("as *")) {
                *finding_id += 1;
                unsafe_inventory.push((entry.to_path_buf(), line_num));
                result.add_finding(
                    Finding::new(
                        format!("BH-UNSAFE-{:04}", finding_id),
                        entry,
                        line_num,
                        "Pointer dereference in unsafe block",
                    )
                    .with_description(format!(
                        "Unsafe block starting at line {}; potential fuzzing target",
                        unsafe_start
                    ))
                    .with_severity(FindingSeverity::High)
                    .with_category(DefectCategory::MemorySafety)
                    .with_suspiciousness(0.75)
                    .with_discovered_by(HuntMode::Fuzz)
                    .with_evidence(FindingEvidence::fuzzing("N/A", "pointer_deref")),
                );
            }

            if line.contains("transmute") {
                *finding_id += 1;
                result.add_finding(
                    Finding::new(
                        format!("BH-UNSAFE-{:04}", finding_id),
                        entry,
                        line_num,
                        "Transmute in unsafe block",
                    )
                    .with_description(
                        "std::mem::transmute bypasses type safety; high-priority fuzzing target",
                    )
                    .with_severity(FindingSeverity::Critical)
                    .with_category(DefectCategory::MemorySafety)
                    .with_suspiciousness(0.9)
                    .with_discovered_by(HuntMode::Fuzz)
                    .with_evidence(FindingEvidence::fuzzing("N/A", "transmute")),
                );
            }
        }

        if line.contains('}') && in_unsafe {
            in_unsafe = false;
        }
    }
}

/// BH-04: Targeted unsafe Rust fuzzing (FourFuzz pattern)
fn run_fuzz_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
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

    let mut unsafe_inventory = Vec::new();
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", target_path.display())) {
            for entry in entries.flatten() {
                scan_file_for_unsafe_blocks(
                    &entry,
                    &mut finding_id,
                    &mut unsafe_inventory,
                    result,
                );
            }
        }
    }

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

    result.stats.mode_stats.fuzz_coverage = if unsafe_inventory.is_empty() {
        100.0
    } else {
        0.0
    };
}

/// Scan a single file for deeply nested conditionals and complex boolean guards.
fn scan_file_for_deep_conditionals(
    entry: &Path,
    finding_id: &mut usize,
    result: &mut HuntResult,
) {
    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let mut complexity: usize = 0;
    let mut complex_start: usize = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;

        if line.contains("if ") || line.contains("match ") {
            complexity += 1;
            if complexity == 1 {
                complex_start = line_num;
            }
        }

        if complexity >= 3 && line.contains("if ") {
            *finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-DEEP-{:04}", *finding_id),
                    entry,
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

        if line.contains(" && ") && line.contains(" || ") {
            *finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-DEEP-{:04}", *finding_id),
                    entry,
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

        if line.contains('}') && complexity > 0 {
            complexity -= 1;
        }
    }
}

/// BH-05: Hybrid concolic + SBFL (COTTONTAIL pattern)
fn run_deep_hunt_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", target_path.display())) {
            for entry in entries.flatten() {
                scan_file_for_deep_conditionals(&entry, &mut finding_id, result);
            }
        }
    }

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

        // File with all three pattern types
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

        // Should find all three types
        assert!(result.findings.len() >= 3, "Expected >= 3 findings, got {}", result.findings.len());

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_bh_mod_009_analyze_mutations_nonexistent_file() {
        let config = HuntConfig::default();
        let mut result = HuntResult::new("/tmp", HuntMode::Falsify, config.clone());

        // Should silently return without panic
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

        // 6 uncovered lines > 5 threshold → should create a finding
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

        // Only 2 uncovered lines ≤ 5 threshold → no findings
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

        // Write a Rust source file with known patterns
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

        // Should find at least unwrap(), TODO, unsafe, panic patterns
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
        // Empty src directory — no files to scan

        let config = HuntConfig {
            targets: vec![PathBuf::from("src")],
            ..Default::default()
        };
        let mut result = HuntResult::new(&temp, HuntMode::Analyze, config.clone());

        analyze_common_patterns(&temp, &config, &mut result);
        // Should complete without panic
        // No files → no findings
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
        // Should complete without panic — no crash logs to find

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

        // Should parse the stack trace and create findings
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
        // Should silently return without panic
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

        // Should only find src/main.rs:5 (not the .cargo path)
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
}
