//! Stack Drift Detection
//!
//! Detects when PAIML stack crates are using outdated versions of other
//! stack crates. This ensures stack coherence and prevents version drift.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Blocks operations when drift detected
//! - **Genchi Genbutsu**: Real-time crates.io dependency verification
//! - **Kaizen**: Continuous improvement through enforced updates

#![allow(dead_code)] // Public API - will be used by stack drift subcommand

use super::crates_io::{CratesIoClient, DependencyData};
use super::{is_paiml_crate, PAIML_CRATES};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity of version drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Major version difference (e.g., 0.10 vs 0.11) - likely breaking
    Major,
    /// Minor version difference within same major (e.g., 0.10.1 vs 0.10.5)
    Minor,
    /// Patch version difference - negligible
    Patch,
}

impl DriftSeverity {
    /// Get display string for severity
    pub fn as_str(&self) -> &'static str {
        match self {
            DriftSeverity::Major => "MAJOR",
            DriftSeverity::Minor => "MINOR",
            DriftSeverity::Patch => "PATCH",
        }
    }
}

impl std::fmt::Display for DriftSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single drift issue detected in the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// The crate that has the outdated dependency
    pub crate_name: String,
    /// Version of the crate with the drift
    pub crate_version: String,
    /// The dependency that is behind
    pub dependency: String,
    /// Version requirement the crate uses
    pub uses_version: String,
    /// Latest version available on crates.io
    pub latest_version: String,
    /// Severity of the drift
    pub severity: DriftSeverity,
}

impl DriftReport {
    /// Format for display
    pub fn display(&self) -> String {
        format!(
            "{} {}: {} {} → {} ({})",
            self.crate_name,
            self.crate_version,
            self.dependency,
            self.uses_version,
            self.latest_version,
            self.severity
        )
    }
}

/// Drift checker for the PAIML stack
pub struct DriftChecker {
    /// Latest versions of PAIML crates (cached)
    latest_versions: HashMap<String, semver::Version>,
}

impl DriftChecker {
    /// Create a new drift checker
    pub fn new() -> Self {
        Self {
            latest_versions: HashMap::new(),
        }
    }

    /// Fetch latest versions of all PAIML crates
    #[cfg(feature = "native")]
    pub async fn fetch_latest_versions(&mut self, client: &mut CratesIoClient) -> Result<()> {
        for crate_name in PAIML_CRATES {
            match client.get_latest_version(crate_name).await {
                Ok(version) => {
                    self.latest_versions
                        .insert((*crate_name).to_string(), version);
                }
                Err(_) => {
                    // Crate not published yet, skip
                    continue;
                }
            }
        }
        Ok(())
    }

    /// Detect drift across the entire stack
    ///
    /// For each published PAIML crate, checks if its dependencies on other
    /// PAIML crates are using the latest versions.
    #[cfg(feature = "native")]
    pub async fn detect_drift(&mut self, client: &mut CratesIoClient) -> Result<Vec<DriftReport>> {
        // First, ensure we have latest versions
        if self.latest_versions.is_empty() {
            self.fetch_latest_versions(client).await?;
        }

        let mut drifts = Vec::new();

        // Check each published PAIML crate
        for crate_name in PAIML_CRATES {
            let crate_version = match self.latest_versions.get(*crate_name) {
                Some(v) => v.to_string(),
                None => continue, // Not published
            };

            // Get dependencies for this crate version
            let deps = match client.get_dependencies(crate_name, &crate_version).await {
                Ok(d) => d,
                Err(_) => continue, // Skip if can't fetch deps
            };

            // Check each dependency
            for dep in deps {
                // Only check PAIML dependencies
                if !is_paiml_crate(&dep.crate_id) {
                    continue;
                }

                // Skip dev dependencies
                if dep.kind == "dev" {
                    continue;
                }

                // Get latest version of this dependency
                let latest = match self.latest_versions.get(&dep.crate_id) {
                    Some(v) => v,
                    None => continue, // Dependency not published
                };

                // Check if behind
                if let Some(drift) = self.check_drift(crate_name, &crate_version, &dep, latest) {
                    drifts.push(drift);
                }
            }
        }

        // Sort by severity (major first) then by crate name
        drifts.sort_by(|a, b| match (&a.severity, &b.severity) {
            (DriftSeverity::Major, DriftSeverity::Major) => a.crate_name.cmp(&b.crate_name),
            (DriftSeverity::Major, _) => std::cmp::Ordering::Less,
            (_, DriftSeverity::Major) => std::cmp::Ordering::Greater,
            (DriftSeverity::Minor, DriftSeverity::Minor) => a.crate_name.cmp(&b.crate_name),
            (DriftSeverity::Minor, _) => std::cmp::Ordering::Less,
            (_, DriftSeverity::Minor) => std::cmp::Ordering::Greater,
            _ => a.crate_name.cmp(&b.crate_name),
        });

        Ok(drifts)
    }

    /// Check if a dependency version is behind the latest
    fn check_drift(
        &self,
        crate_name: &str,
        crate_version: &str,
        dep: &DependencyData,
        latest: &semver::Version,
    ) -> Option<DriftReport> {
        // Parse the version requirement to extract the base version
        let uses_version = &dep.version_req;

        // Try to parse a semver from the requirement
        // Handle common patterns: "0.11", "0.11.0", "^0.11", "~0.11"
        let version_str = uses_version
            .trim_start_matches('^')
            .trim_start_matches('~')
            .trim_start_matches('=')
            .trim_start_matches('>')
            .trim_start_matches('<')
            .trim();

        // Parse as version or partial version
        let (uses_major, uses_minor) = Self::parse_version_parts(version_str);

        // Compare with latest
        let severity = if uses_major < latest.major as u32 {
            // Major version behind (rare for 0.x crates)
            Some(DriftSeverity::Major)
        } else if uses_major == latest.major as u32 && uses_minor < latest.minor as u32 {
            // Minor version behind within same major
            Some(DriftSeverity::Major) // For 0.x, minor is effectively major
        } else {
            // Up to date or ahead
            None
        };

        severity.map(|sev| DriftReport {
            crate_name: crate_name.to_string(),
            crate_version: crate_version.to_string(),
            dependency: dep.crate_id.clone(),
            uses_version: uses_version.clone(),
            latest_version: latest.to_string(),
            severity: sev,
        })
    }

    /// Parse version string into (major, minor) parts
    fn parse_version_parts(version_str: &str) -> (u32, u32) {
        let parts: Vec<&str> = version_str.split('.').collect();
        let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        (major, minor)
    }

    /// Get cached latest versions
    pub fn latest_versions(&self) -> &HashMap<String, semver::Version> {
        &self.latest_versions
    }
}

impl Default for DriftChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Format drift reports for display (blocking error style)
pub fn format_drift_errors(drifts: &[DriftReport]) -> String {
    if drifts.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    output.push_str("🔴 Stack Drift Detected - Cannot Proceed\n\n");

    for drift in drifts {
        output.push_str(&format!("   {}\n", drift.display()));
    }

    output.push_str("\nStack drift detected. Fix dependencies before proceeding.\n");
    output.push_str("Run: batuta stack drift --fix\n");
    output.push_str("Or use --allow-drift to bypass (for local development).\n");

    output
}

/// Format drift reports as JSON
pub fn format_drift_json(drifts: &[DriftReport]) -> Result<String> {
    Ok(serde_json::to_string_pretty(drifts)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_severity_display() {
        assert_eq!(DriftSeverity::Major.as_str(), "MAJOR");
        assert_eq!(DriftSeverity::Minor.as_str(), "MINOR");
        assert_eq!(DriftSeverity::Patch.as_str(), "PATCH");
    }

    #[test]
    fn test_drift_report_display() {
        let report = DriftReport {
            crate_name: "trueno-rag".to_string(),
            crate_version: "0.1.5".to_string(),
            dependency: "trueno".to_string(),
            uses_version: "0.10.1".to_string(),
            latest_version: "0.11.0".to_string(),
            severity: DriftSeverity::Major,
        };

        let display = report.display();
        assert!(display.contains("trueno-rag"));
        assert!(display.contains("trueno"));
        assert!(display.contains("0.10.1"));
        assert!(display.contains("0.11.0"));
        assert!(display.contains("MAJOR"));
    }

    #[test]
    fn test_parse_version_parts() {
        assert_eq!(DriftChecker::parse_version_parts("0.11.0"), (0, 11));
        assert_eq!(DriftChecker::parse_version_parts("0.11"), (0, 11));
        assert_eq!(DriftChecker::parse_version_parts("1.2.3"), (1, 2));
        assert_eq!(DriftChecker::parse_version_parts("2"), (2, 0));
    }

    #[test]
    fn test_format_drift_errors_empty() {
        let output = format_drift_errors(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_drift_errors_with_drifts() {
        let drifts = vec![DriftReport {
            crate_name: "trueno-rag".to_string(),
            crate_version: "0.1.5".to_string(),
            dependency: "trueno".to_string(),
            uses_version: "0.10.1".to_string(),
            latest_version: "0.11.0".to_string(),
            severity: DriftSeverity::Major,
        }];

        let output = format_drift_errors(&drifts);
        assert!(output.contains("Stack Drift Detected"));
        assert!(output.contains("trueno-rag"));
        assert!(output.contains("batuta stack drift --fix"));
    }

    #[test]
    fn test_drift_checker_new() {
        let checker = DriftChecker::new();
        assert!(checker.latest_versions.is_empty());
    }

    #[test]
    fn test_drift_checker_default() {
        let checker = DriftChecker::default();
        assert!(checker.latest_versions.is_empty());
    }
}
