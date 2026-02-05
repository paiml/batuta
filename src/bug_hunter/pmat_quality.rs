//! PMAT Quality Integration for Bug Hunter (BH-21 to BH-25)
//!
//! Integrates PMAT function-level quality metrics into the bug hunting pipeline:
//! - BH-21: Quality-weighted suspiciousness scoring
//! - BH-22: Smart target scoping by quality
//! - BH-23: SATD-enriched findings from PMAT data
//! - BH-24: Regression risk scoring
//! - BH-25: Spec claim quality gates

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{DefectCategory, Finding, FindingEvidence, FindingSeverity, HuntMode};
use crate::tools;

/// Lightweight PMAT query result for bug-hunter integration.
///
/// Mirrors the fields we need from `pmat query --format json` output,
/// avoiding cross-crate dependency on the CLI module.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PmatFunctionInfo {
    pub file_path: String,
    pub function_name: String,
    #[serde(default)]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: usize,
    #[serde(default)]
    pub tdg_score: f64,
    #[serde(default)]
    pub tdg_grade: String,
    #[serde(default)]
    pub complexity: u32,
    #[serde(default)]
    pub satd_count: u32,
}

/// Quality index: maps file paths to their PMAT function-level results.
pub type PmatQualityIndex = HashMap<PathBuf, Vec<PmatFunctionInfo>>;

// ============================================================================
// Availability
// ============================================================================

/// Check if PMAT tool is available on the system.
pub fn pmat_available() -> bool {
    tools::ToolRegistry::detect().pmat.is_some()
}

// ============================================================================
// Index construction
// ============================================================================

/// Run `pmat query` and parse the JSON output.
fn run_pmat_query_raw(
    project_path: &Path,
    query: &str,
    limit: usize,
) -> Option<Vec<PmatFunctionInfo>> {
    let limit_str = limit.to_string();
    let args: Vec<&str> = vec!["query", query, "--format", "json", "--limit", &limit_str];
    let output = tools::run_tool("pmat", &args, Some(project_path)).ok()?;
    let results: Vec<PmatFunctionInfo> = serde_json::from_str(&output).ok()?;
    Some(results)
}

/// Build a quality index by running `pmat query` against the project.
///
/// Returns `None` if pmat is unavailable or the query returns no results.
pub fn build_quality_index(
    project_path: &Path,
    query: &str,
    limit: usize,
) -> Option<PmatQualityIndex> {
    if !pmat_available() {
        return None;
    }

    let results = run_pmat_query_raw(project_path, query, limit)?;
    if results.is_empty() {
        return None;
    }

    Some(index_from_results(results))
}

/// Group PMAT query results by file path, sorting each group by start_line.
pub fn index_from_results(results: Vec<PmatFunctionInfo>) -> PmatQualityIndex {
    let mut index: PmatQualityIndex = HashMap::new();
    for result in results {
        let path = PathBuf::from(&result.file_path);
        index.entry(path).or_default().push(result);
    }
    // Sort each file's functions by start_line
    for functions in index.values_mut() {
        functions.sort_by_key(|f: &PmatFunctionInfo| f.start_line);
    }
    index
}

/// Look up the PMAT result for a specific file and line number.
///
/// First tries exact span match (start_line <= line <= end_line),
/// then falls back to nearest function by start_line.
pub fn lookup_quality<'a>(
    index: &'a PmatQualityIndex,
    file: &Path,
    line: usize,
) -> Option<&'a PmatFunctionInfo> {
    let functions = index.get(file)?;

    // Exact span match
    if let Some(f) = functions
        .iter()
        .find(|f| line >= f.start_line && line <= f.end_line)
    {
        return Some(f);
    }

    // Fallback: nearest function by start_line
    functions
        .iter()
        .min_by_key(|f| (f.start_line as isize - line as isize).unsigned_abs())
}

// ============================================================================
// BH-21: Quality-adjusted suspiciousness
// ============================================================================

/// Adjust a base suspiciousness score using TDG quality data.
///
/// Formula: `(base * (1 + weight * (0.5 - tdg/100))).clamp(0, 1)`
///
/// TDG 50 = baseline (no change). Low-quality code (TDG < 50) gets boosted
/// suspiciousness; high-quality code (TDG > 50) gets reduced suspiciousness.
pub fn quality_adjusted_suspiciousness(base: f64, tdg_score: f64, weight: f64) -> f64 {
    let quality_factor = 0.5 - (tdg_score / 100.0);
    (base * (1.0 + weight * quality_factor)).clamp(0.0, 1.0)
}

/// Apply quality weights to findings in-place, adding quality evidence.
pub fn apply_quality_weights(findings: &mut [Finding], index: &PmatQualityIndex, weight: f64) {
    for finding in findings.iter_mut() {
        if let Some(pmat) = lookup_quality(index, &finding.file, finding.line) {
            let original = finding.suspiciousness;
            finding.suspiciousness =
                quality_adjusted_suspiciousness(original, pmat.tdg_score, weight);
            finding.evidence.push(FindingEvidence::quality_metrics(
                &pmat.tdg_grade,
                pmat.tdg_score,
                pmat.complexity,
            ));
        }
    }
}

// ============================================================================
// BH-24: Regression risk scoring
// ============================================================================

/// Compute regression risk from PMAT quality data.
///
/// Formula: `0.5 * (1 - tdg/100) + 0.3 * (complexity/50).min(1) + 0.2 * (satd/5).min(1)`
///
/// Returns a value in [0.0, 1.0] where higher means greater regression risk.
pub fn compute_regression_risk(pmat: &PmatFunctionInfo) -> f64 {
    let tdg_factor = 1.0 - (pmat.tdg_score / 100.0);
    let cx_factor = (pmat.complexity as f64 / 50.0).min(1.0);
    let satd_factor = (pmat.satd_count as f64 / 5.0).min(1.0);
    let risk: f64 = 0.5 * tdg_factor + 0.3 * cx_factor + 0.2 * satd_factor;
    risk.clamp(0.0, 1.0)
}

/// Apply regression risk scores to findings in-place.
pub fn apply_regression_risk(findings: &mut [Finding], index: &PmatQualityIndex) {
    for finding in findings.iter_mut() {
        if let Some(pmat) = lookup_quality(index, &finding.file, finding.line) {
            finding.regression_risk = Some(compute_regression_risk(pmat));
        }
    }
}

// ============================================================================
// BH-22: Smart target scoping
// ============================================================================

/// Scope targets to files with worst quality, returning paths sorted by TDG ascending.
#[allow(dead_code)]
pub fn scope_targets_by_quality(
    project_path: &Path,
    query: &str,
    limit: usize,
) -> Option<Vec<PathBuf>> {
    if !pmat_available() {
        return None;
    }

    let results = run_pmat_query_raw(project_path, query, limit)?;
    if results.is_empty() {
        return None;
    }

    // Group by file, compute average TDG per file
    let mut file_scores: HashMap<PathBuf, (f64, usize)> = HashMap::new();
    for r in &results {
        let path = PathBuf::from(&r.file_path);
        let entry = file_scores.entry(path).or_insert((0.0, 0));
        entry.0 += r.tdg_score;
        entry.1 += 1;
    }

    // Sort by average TDG ascending (worst quality first)
    let mut files: Vec<(PathBuf, f64)> = file_scores
        .into_iter()
        .map(|(path, (sum, count))| (path, sum / count as f64))
        .collect();
    files.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(files.into_iter().map(|(path, _)| path).collect())
}

// ============================================================================
// BH-23: SATD-enriched findings
// ============================================================================

/// Generate findings from PMAT SATD (self-admitted technical debt) data.
pub fn generate_satd_findings(_project_path: &Path, index: &PmatQualityIndex) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut id_counter = 0u32;

    for (file, functions) in index {
        for func in functions {
            if func.satd_count > 0 {
                id_counter += 1;
                let severity = match func.satd_count {
                    1 => FindingSeverity::Low,
                    2..=3 => FindingSeverity::Medium,
                    _ => FindingSeverity::High,
                };
                let suspiciousness = match func.satd_count {
                    1 => 0.3,
                    2..=3 => 0.5,
                    _ => 0.7,
                };
                findings.push(
                    Finding::new(
                        format!("BH-SATD-{:04}", id_counter),
                        file,
                        func.start_line,
                        format!(
                            "PMAT: {} SATD markers in `{}`",
                            func.satd_count, func.function_name
                        ),
                    )
                    .with_description(format!(
                        "Function `{}` (grade {}, complexity {}) has {} self-admitted technical debt markers",
                        func.function_name, func.tdg_grade, func.complexity, func.satd_count
                    ))
                    .with_severity(severity)
                    .with_category(DefectCategory::LogicErrors)
                    .with_suspiciousness(suspiciousness)
                    .with_discovered_by(HuntMode::Analyze)
                    .with_evidence(FindingEvidence::quality_metrics(
                        &func.tdg_grade,
                        func.tdg_score,
                        func.complexity,
                    )),
                );
            }
        }
    }

    findings
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BH-PMAT-001: pmat_available
    // =========================================================================

    #[test]
    fn test_pmat_available_returns_bool() {
        let _ = pmat_available();
    }

    // =========================================================================
    // BH-PMAT-002: quality_adjusted_suspiciousness
    // =========================================================================

    #[test]
    fn test_quality_adjusted_high_quality_reduces() {
        let adjusted = quality_adjusted_suspiciousness(0.5, 95.0, 0.5);
        assert!(
            adjusted < 0.5,
            "High quality should reduce suspiciousness: got {}",
            adjusted
        );
    }

    #[test]
    fn test_quality_adjusted_low_quality_boosts() {
        let adjusted = quality_adjusted_suspiciousness(0.5, 20.0, 0.5);
        assert!(
            adjusted > 0.5,
            "Low quality should boost suspiciousness: got {}",
            adjusted
        );
    }

    #[test]
    fn test_quality_adjusted_baseline_no_change() {
        // TDG 50 is baseline, quality_factor = 0, so no change
        let adjusted = quality_adjusted_suspiciousness(0.5, 50.0, 0.5);
        assert!((adjusted - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_quality_adjusted_clamping() {
        let adjusted = quality_adjusted_suspiciousness(0.9, 10.0, 1.0);
        assert!(adjusted <= 1.0, "Should clamp to 1.0: got {}", adjusted);

        let adjusted = quality_adjusted_suspiciousness(0.0, 10.0, 1.0);
        assert!((adjusted - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_adjusted_weight_zero() {
        let adjusted = quality_adjusted_suspiciousness(0.5, 20.0, 0.0);
        assert!((adjusted - 0.5).abs() < 0.001);
    }

    // =========================================================================
    // BH-PMAT-003: index_from_results
    // =========================================================================

    #[test]
    fn test_index_from_results_groups_by_file() {
        let results = vec![
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "alpha".into(),
                start_line: 10,
                end_line: 20,
                tdg_score: 80.0,
                tdg_grade: "B".into(),
                complexity: 5,
                satd_count: 0,
            },
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "beta".into(),
                start_line: 30,
                end_line: 50,
                tdg_score: 60.0,
                tdg_grade: "C".into(),
                complexity: 12,
                satd_count: 2,
            },
            PmatFunctionInfo {
                file_path: "src/main.rs".into(),
                function_name: "main".into(),
                start_line: 1,
                end_line: 10,
                tdg_score: 95.0,
                tdg_grade: "A".into(),
                complexity: 2,
                satd_count: 0,
            },
        ];

        let index = index_from_results(results);
        assert_eq!(index.len(), 2);
        assert_eq!(index[&PathBuf::from("src/lib.rs")].len(), 2);
        assert_eq!(index[&PathBuf::from("src/main.rs")].len(), 1);
        let lib_fns = &index[&PathBuf::from("src/lib.rs")];
        assert!(lib_fns[0].start_line < lib_fns[1].start_line);
    }

    // =========================================================================
    // BH-PMAT-004: lookup_quality
    // =========================================================================

    #[test]
    fn test_lookup_quality_exact_span() {
        let results = vec![PmatFunctionInfo {
            file_path: "src/lib.rs".into(),
            function_name: "process".into(),
            start_line: 10,
            end_line: 30,
            tdg_score: 75.0,
            tdg_grade: "C".into(),
            complexity: 0,
            satd_count: 0,
        }];
        let index = index_from_results(results);

        let result = lookup_quality(&index, Path::new("src/lib.rs"), 20);
        assert!(result.is_some());
        assert_eq!(result.unwrap().function_name, "process");
    }

    #[test]
    fn test_lookup_quality_nearest_fallback() {
        let results = vec![
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "alpha".into(),
                start_line: 10,
                end_line: 20,
                tdg_score: 0.0,
                tdg_grade: String::new(),
                complexity: 0,
                satd_count: 0,
            },
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "beta".into(),
                start_line: 50,
                end_line: 70,
                tdg_score: 0.0,
                tdg_grade: String::new(),
                complexity: 0,
                satd_count: 0,
            },
        ];
        let index = index_from_results(results);

        let result = lookup_quality(&index, Path::new("src/lib.rs"), 35);
        assert!(result.is_some());
    }

    #[test]
    fn test_lookup_quality_no_file() {
        let index: PmatQualityIndex = HashMap::new();
        let result = lookup_quality(&index, Path::new("nonexistent.rs"), 10);
        assert!(result.is_none());
    }

    // =========================================================================
    // BH-PMAT-005: compute_regression_risk
    // =========================================================================

    #[test]
    fn test_regression_risk_high_quality() {
        let pmat = PmatFunctionInfo {
            file_path: String::new(),
            function_name: String::new(),
            start_line: 0,
            end_line: 0,
            tdg_score: 95.0,
            tdg_grade: "A".into(),
            complexity: 3,
            satd_count: 0,
        };
        let risk = compute_regression_risk(&pmat);
        assert!(risk < 0.1, "High quality should have low risk: got {}", risk);
    }

    #[test]
    fn test_regression_risk_low_quality() {
        let pmat = PmatFunctionInfo {
            file_path: String::new(),
            function_name: String::new(),
            start_line: 0,
            end_line: 0,
            tdg_score: 20.0,
            tdg_grade: "F".into(),
            complexity: 40,
            satd_count: 8,
        };
        let risk = compute_regression_risk(&pmat);
        assert!(risk > 0.7, "Low quality should have high risk: got {}", risk);
    }

    #[test]
    fn test_regression_risk_clamped() {
        let pmat = PmatFunctionInfo {
            file_path: String::new(),
            function_name: String::new(),
            start_line: 0,
            end_line: 0,
            tdg_score: 0.0,
            tdg_grade: "F".into(),
            complexity: 100,
            satd_count: 20,
        };
        let risk = compute_regression_risk(&pmat);
        assert!(risk <= 1.0);
        assert!(risk >= 0.0);
    }

    // =========================================================================
    // BH-PMAT-006: generate_satd_findings
    // =========================================================================

    #[test]
    fn test_generate_satd_findings_produces_findings() {
        let results = vec![
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "messy_fn".into(),
                start_line: 10,
                end_line: 50,
                tdg_score: 40.0,
                tdg_grade: "D".into(),
                complexity: 15,
                satd_count: 3,
            },
            PmatFunctionInfo {
                file_path: "src/lib.rs".into(),
                function_name: "clean_fn".into(),
                start_line: 60,
                end_line: 70,
                tdg_score: 95.0,
                tdg_grade: "A".into(),
                complexity: 2,
                satd_count: 0,
            },
        ];
        let index = index_from_results(results);
        let findings = generate_satd_findings(Path::new("."), &index);

        assert_eq!(
            findings.len(),
            1,
            "Should only generate for functions with SATD"
        );
        assert!(findings[0].id.starts_with("BH-SATD-"));
        assert!(findings[0].title.contains("messy_fn"));
        assert_eq!(findings[0].severity, FindingSeverity::Medium);
    }

    #[test]
    fn test_generate_satd_findings_severity_scaling() {
        let results = vec![
            PmatFunctionInfo {
                file_path: "a.rs".into(),
                function_name: "low".into(),
                start_line: 1,
                end_line: 10,
                tdg_score: 0.0,
                tdg_grade: String::new(),
                complexity: 0,
                satd_count: 1,
            },
            PmatFunctionInfo {
                file_path: "b.rs".into(),
                function_name: "high".into(),
                start_line: 1,
                end_line: 10,
                tdg_score: 0.0,
                tdg_grade: String::new(),
                complexity: 0,
                satd_count: 5,
            },
        ];
        let index = index_from_results(results);
        let findings = generate_satd_findings(Path::new("."), &index);

        assert_eq!(findings.len(), 2);
        let low_finding = findings.iter().find(|f| f.title.contains("low")).unwrap();
        let high_finding = findings.iter().find(|f| f.title.contains("high")).unwrap();
        assert_eq!(low_finding.severity, FindingSeverity::Low);
        assert_eq!(high_finding.severity, FindingSeverity::High);
    }

    #[test]
    fn test_generate_satd_findings_empty_index() {
        let index: PmatQualityIndex = HashMap::new();
        let findings = generate_satd_findings(Path::new("."), &index);
        assert!(findings.is_empty());
    }

    // =========================================================================
    // BH-PMAT-007: apply_quality_weights
    // =========================================================================

    #[test]
    fn test_apply_quality_weights_adjusts_scores() {
        let results = vec![PmatFunctionInfo {
            file_path: "src/lib.rs".into(),
            function_name: "buggy".into(),
            start_line: 10,
            end_line: 30,
            tdg_score: 30.0,
            tdg_grade: "D".into(),
            complexity: 20,
            satd_count: 0,
        }];
        let index = index_from_results(results);

        let mut findings = vec![Finding::new("F-001", "src/lib.rs", 15, "Test finding")
            .with_suspiciousness(0.5)];

        apply_quality_weights(&mut findings, &index, 0.5);

        assert!(findings[0].suspiciousness > 0.5);
        assert!(findings[0]
            .evidence
            .iter()
            .any(|e| matches!(e.evidence_type, crate::bug_hunter::types::EvidenceKind::QualityMetrics)));
    }

    // =========================================================================
    // BH-PMAT-008: apply_regression_risk
    // =========================================================================

    #[test]
    fn test_apply_regression_risk_sets_risk() {
        let results = vec![PmatFunctionInfo {
            file_path: "src/lib.rs".into(),
            function_name: "risky".into(),
            start_line: 10,
            end_line: 30,
            tdg_score: 30.0,
            tdg_grade: "D".into(),
            complexity: 25,
            satd_count: 4,
        }];
        let index = index_from_results(results);

        let mut findings = vec![Finding::new("F-001", "src/lib.rs", 15, "Test finding")];

        apply_regression_risk(&mut findings, &index);
        assert!(findings[0].regression_risk.is_some());
        assert!(findings[0].regression_risk.unwrap() > 0.3);
    }
}
