//! Bug Hunter Diff Mode
//!
//! Compare findings against a baseline to show only new issues.

use super::types::{Finding, HuntResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

/// Baseline storage for diff comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    /// Git commit hash when baseline was created
    pub commit: String,
    /// Timestamp when baseline was created
    pub timestamp: u64,
    /// Finding fingerprints (file:line:pattern hashes)
    pub fingerprints: HashSet<String>,
}

impl Baseline {
    /// Create a new baseline from findings.
    pub fn from_findings(findings: &[Finding]) -> Self {
        let fingerprints = findings
            .iter()
            .map(fingerprint)
            .collect();

        let commit = get_current_commit().unwrap_or_default();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            commit,
            timestamp,
            fingerprints,
        }
    }

    /// Load baseline from disk.
    pub fn load(project_path: &Path) -> Option<Self> {
        let baseline_path = project_path.join(".pmat").join("bug-hunter-baseline.json");
        if baseline_path.exists() {
            let content = std::fs::read_to_string(&baseline_path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save baseline to disk.
    pub fn save(&self, project_path: &Path) -> Result<(), String> {
        let pmat_dir = project_path.join(".pmat");
        std::fs::create_dir_all(&pmat_dir)
            .map_err(|e| format!("Failed to create .pmat directory: {}", e))?;

        let baseline_path = pmat_dir.join("bug-hunter-baseline.json");
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize baseline: {}", e))?;

        std::fs::write(&baseline_path, content)
            .map_err(|e| format!("Failed to write baseline: {}", e))?;

        Ok(())
    }

    /// Check if a finding is new (not in baseline).
    pub fn is_new(&self, finding: &Finding) -> bool {
        !self.fingerprints.contains(&fingerprint(finding))
    }
}

/// Generate a fingerprint for a finding.
fn fingerprint(finding: &Finding) -> String {
    // Use file path (relative), line number, and title for fingerprinting
    // This allows findings to persist across minor code movements
    let file_name = finding
        .file
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    format!("{}:{}:{}", file_name, finding.line, finding.title)
}

/// Get current git commit hash.
fn get_current_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Get files changed since a commit or time period.
pub fn get_changed_files(project_path: &Path, base: Option<&str>, since: Option<&str>) -> Vec<String> {
    let output = if let Some(base) = base {
        Command::new("git")
            .current_dir(project_path)
            .args(["diff", "--name-only", base])
            .output()
    } else if let Some(since) = since {
        let git_since = format!("--since={}", since);
        Command::new("git")
            .current_dir(project_path)
            .args(["log", "--name-only", "--pretty=format:"])
            .arg(&git_since)
            .output()
    } else {
        return Vec::new();
    };

    match output {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Filter findings to only include new ones.
pub fn filter_new_findings(result: &HuntResult, baseline: &Baseline) -> Vec<Finding> {
    result
        .findings
        .iter()
        .filter(|f| baseline.is_new(f))
        .cloned()
        .collect()
}

/// Filter findings to only those in changed files.
pub fn filter_changed_files(findings: &[Finding], changed_files: &[String]) -> Vec<Finding> {
    findings
        .iter()
        .filter(|f| {
            let file_path = f.file.to_string_lossy();
            changed_files.iter().any(|cf| file_path.ends_with(cf))
        })
        .cloned()
        .collect()
}

/// Diff result showing new and resolved findings.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Newly introduced findings
    pub new_findings: Vec<Finding>,
    /// Findings that were in baseline but not in current
    pub resolved_count: usize,
    /// Total findings in current run
    pub total_current: usize,
    /// Total findings in baseline
    pub total_baseline: usize,
    /// Base commit/time used for comparison
    pub base_reference: String,
}

impl DiffResult {
    /// Create a diff result from current findings and baseline.
    pub fn compute(current: &HuntResult, baseline: &Baseline, base_ref: &str) -> Self {
        let new_findings = filter_new_findings(current, baseline);
        let current_fps: HashSet<String> = current.findings.iter().map(fingerprint).collect();

        let resolved_count = baseline
            .fingerprints
            .iter()
            .filter(|fp| !current_fps.contains(*fp))
            .count();

        Self {
            new_findings,
            resolved_count,
            total_current: current.findings.len(),
            total_baseline: baseline.fingerprints.len(),
            base_reference: base_ref.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bug_hunter::types::{DefectCategory, FindingSeverity};
    use std::path::PathBuf;

    fn make_finding(file: &str, line: usize, title: &str) -> Finding {
        Finding::new(
            "TEST-001".to_string(),
            &PathBuf::from(file),
            line,
            title.to_string(),
        )
        .with_severity(FindingSeverity::Medium)
        .with_category(DefectCategory::LogicErrors)
        .with_suspiciousness(0.5)
    }

    #[test]
    fn test_fingerprint() {
        let f = make_finding("src/foo.rs", 42, "Pattern: TODO");
        let fp = fingerprint(&f);
        assert!(fp.contains("foo.rs"));
        assert!(fp.contains("42"));
        assert!(fp.contains("TODO"));
    }

    #[test]
    fn test_baseline_is_new() {
        let findings = vec![
            make_finding("src/foo.rs", 10, "Pattern: TODO"),
            make_finding("src/bar.rs", 20, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&findings);

        // Same finding is not new
        let same = make_finding("src/foo.rs", 10, "Pattern: TODO");
        assert!(!baseline.is_new(&same));

        // Different finding is new
        let new = make_finding("src/baz.rs", 30, "Pattern: HACK");
        assert!(baseline.is_new(&new));

        // Same file, different line is new
        let new_line = make_finding("src/foo.rs", 15, "Pattern: TODO");
        assert!(baseline.is_new(&new_line));
    }

    #[test]
    fn test_diff_result() {
        let baseline_findings = vec![
            make_finding("src/foo.rs", 10, "Pattern: TODO"),
            make_finding("src/bar.rs", 20, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&baseline_findings);

        let current = HuntResult {
            findings: vec![
                make_finding("src/foo.rs", 10, "Pattern: TODO"), // existing
                make_finding("src/baz.rs", 30, "Pattern: HACK"), // new
            ],
            ..Default::default()
        };

        let diff = DiffResult::compute(&current, &baseline, "main");

        assert_eq!(diff.new_findings.len(), 1);
        assert_eq!(diff.resolved_count, 1); // bar.rs:20 was resolved
        assert_eq!(diff.total_current, 2);
        assert_eq!(diff.total_baseline, 2);
    }
}
