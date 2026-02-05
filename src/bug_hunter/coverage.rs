//! Coverage-Based Hotpath Weighting
//!
//! Applies coverage data to weight findings - bugs in uncovered code paths
//! are more suspicious than bugs in well-covered code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{EvidenceKind, Finding, FindingEvidence};

/// Coverage index: maps (file, line) to execution count.
pub type CoverageIndex = HashMap<(PathBuf, usize), usize>;

/// Parse LCOV file into a coverage index.
pub fn parse_lcov(content: &str) -> CoverageIndex {
    let mut index = CoverageIndex::new();
    let mut current_file: Option<PathBuf> = None;

    for line in content.lines() {
        if let Some(file) = line.strip_prefix("SF:") {
            current_file = Some(PathBuf::from(file.trim()));
        } else if let Some(da) = line.strip_prefix("DA:") {
            if let Some(ref file) = current_file {
                let parts: Vec<&str> = da.split(',').collect();
                if parts.len() >= 2 {
                    if let (Ok(line_num), Ok(hits)) =
                        (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                    {
                        index.insert((file.clone(), line_num), hits);
                    }
                }
            }
        } else if line == "end_of_record" {
            current_file = None;
        }
    }

    index
}

/// Load coverage index from an LCOV file.
pub fn load_coverage_index(lcov_path: &Path) -> Option<CoverageIndex> {
    let content = std::fs::read_to_string(lcov_path).ok()?;
    Some(parse_lcov(&content))
}

/// Find the best coverage file from standard locations.
pub fn find_coverage_file(project_path: &Path) -> Option<PathBuf> {
    let candidates = [
        project_path.join("lcov.info"),
        project_path.join("target/coverage/lcov.info"),
        project_path.join("coverage/lcov.info"),
        project_path.join("target/llvm-cov/lcov.info"),
    ];

    candidates.into_iter().find(|c| c.exists())
}

/// Look up coverage for a specific file and line.
///
/// Returns the execution count, or None if the line isn't in the coverage data.
pub fn lookup_coverage(index: &CoverageIndex, file: &Path, line: usize) -> Option<usize> {
    // Try exact match first
    if let Some(&hits) = index.get(&(file.to_path_buf(), line)) {
        return Some(hits);
    }

    // Try matching just the filename for relative/absolute path differences
    let file_name = file.file_name()?.to_string_lossy();
    for ((path, l), &hits) in index.iter() {
        if *l == line {
            if let Some(name) = path.file_name() {
                if name.to_string_lossy() == file_name {
                    return Some(hits);
                }
            }
        }
    }

    None
}

/// Compute coverage weight factor for a hit count.
///
/// Returns a value in [-0.5, 0.5]:
/// - Uncovered (0 hits): +0.5 (boost suspiciousness)
/// - Low coverage (1-5 hits): +0.2
/// - Medium coverage (6-20 hits): 0.0 (neutral)
/// - High coverage (>20 hits): -0.3 (reduce suspiciousness)
fn coverage_factor(hits: usize) -> f64 {
    match hits {
        0 => 0.5,
        1..=5 => 0.2,
        6..=20 => 0.0,
        _ => -0.3,
    }
}

/// Adjust suspiciousness based on coverage.
///
/// Formula: `(base * (1 + weight * coverage_factor)).clamp(0, 1)`
pub fn coverage_adjusted_suspiciousness(base: f64, hits: usize, weight: f64) -> f64 {
    let factor = coverage_factor(hits);
    (base * (1.0 + weight * factor)).clamp(0.0, 1.0)
}

/// Apply coverage weights to findings in-place.
pub fn apply_coverage_weights(findings: &mut [Finding], index: &CoverageIndex, weight: f64) {
    for finding in findings.iter_mut() {
        if let Some(hits) = lookup_coverage(index, &finding.file, finding.line) {
            let original = finding.suspiciousness;
            finding.suspiciousness = coverage_adjusted_suspiciousness(original, hits, weight);

            // Add evidence
            let coverage_desc = match hits {
                0 => "uncovered".to_string(),
                n => format!("{} hits", n),
            };
            finding.evidence.push(FindingEvidence {
                evidence_type: EvidenceKind::SbflScore,
                description: format!("Coverage: {}", coverage_desc),
                data: Some(hits.to_string()),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lcov_basic() {
        let content = r#"SF:src/lib.rs
DA:1,10
DA:2,5
DA:3,0
end_of_record
SF:src/main.rs
DA:1,1
end_of_record
"#;

        let index = parse_lcov(content);
        assert_eq!(index.len(), 4);
        assert_eq!(index.get(&(PathBuf::from("src/lib.rs"), 1)), Some(&10));
        assert_eq!(index.get(&(PathBuf::from("src/lib.rs"), 3)), Some(&0));
        assert_eq!(index.get(&(PathBuf::from("src/main.rs"), 1)), Some(&1));
    }

    #[test]
    fn test_parse_lcov_empty() {
        let index = parse_lcov("");
        assert!(index.is_empty());
    }

    #[test]
    fn test_coverage_factor() {
        assert_eq!(coverage_factor(0), 0.5); // uncovered = boost
        assert_eq!(coverage_factor(3), 0.2); // low coverage = small boost
        assert_eq!(coverage_factor(10), 0.0); // medium = neutral
        assert_eq!(coverage_factor(100), -0.3); // high = reduce
    }

    #[test]
    fn test_coverage_adjusted_suspiciousness() {
        // Uncovered code gets boosted
        let adjusted = coverage_adjusted_suspiciousness(0.5, 0, 1.0);
        assert!(adjusted > 0.5);
        assert!((adjusted - 0.75).abs() < 0.01);

        // High coverage gets reduced
        let adjusted = coverage_adjusted_suspiciousness(0.5, 100, 1.0);
        assert!(adjusted < 0.5);
        assert!((adjusted - 0.35).abs() < 0.01);

        // Neutral coverage unchanged
        let adjusted = coverage_adjusted_suspiciousness(0.5, 10, 1.0);
        assert!((adjusted - 0.5).abs() < 0.01);

        // Zero weight = no change
        let adjusted = coverage_adjusted_suspiciousness(0.5, 0, 0.0);
        assert!((adjusted - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_lookup_coverage_exact() {
        let mut index = CoverageIndex::new();
        index.insert((PathBuf::from("src/lib.rs"), 10), 5);

        let result = lookup_coverage(&index, Path::new("src/lib.rs"), 10);
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_lookup_coverage_filename_match() {
        let mut index = CoverageIndex::new();
        index.insert((PathBuf::from("/full/path/to/lib.rs"), 10), 5);

        // Should match by filename even with different path
        let result = lookup_coverage(&index, Path::new("src/lib.rs"), 10);
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_lookup_coverage_not_found() {
        let index = CoverageIndex::new();
        let result = lookup_coverage(&index, Path::new("src/lib.rs"), 10);
        assert_eq!(result, None);
    }

    #[test]
    fn test_apply_coverage_weights() {
        use crate::bug_hunter::Finding;

        let mut index = CoverageIndex::new();
        index.insert((PathBuf::from("src/lib.rs"), 10), 0); // uncovered

        let mut findings = vec![Finding::new("F-001", "src/lib.rs", 10, "Test")
            .with_suspiciousness(0.5)];

        apply_coverage_weights(&mut findings, &index, 1.0);

        assert!(findings[0].suspiciousness > 0.5);
        assert!(findings[0]
            .evidence
            .iter()
            .any(|e| e.description.contains("Coverage")));
    }
}
