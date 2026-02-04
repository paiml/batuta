//! Bug Hunter Types
//!
//! Types for representing bug hunting results, findings, and hunt configurations.
//!
//! # Popperian Philosophy
//!
//! Bug hunting operationalizes falsification: we systematically attempt to break
//! code, not merely verify it works. Each finding represents a successful
//! falsification of the implicit claim "this code is correct."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Mode of bug hunting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HuntMode {
    /// Mutation-based invariant falsification (FDV pattern)
    Falsify,
    /// SBFL without failing tests (SBEST pattern)
    Hunt,
    /// LLM-augmented static analysis (LLIFT pattern)
    Analyze,
    /// Targeted unsafe Rust fuzzing (FourFuzz pattern)
    Fuzz,
    /// Hybrid concolic + SBFL (COTTONTAIL pattern)
    DeepHunt,
}

impl std::fmt::Display for HuntMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HuntMode::Falsify => write!(f, "Falsify"),
            HuntMode::Hunt => write!(f, "Hunt"),
            HuntMode::Analyze => write!(f, "Analyze"),
            HuntMode::Fuzz => write!(f, "Fuzz"),
            HuntMode::DeepHunt => write!(f, "DeepHunt"),
        }
    }
}

/// Severity of a bug finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FindingSeverity {
    /// Informational finding
    Info,
    /// Low severity - style or minor issue
    Low,
    /// Medium severity - potential bug
    Medium,
    /// High severity - likely bug
    High,
    /// Critical - security vulnerability or crash
    Critical,
}

impl std::fmt::Display for FindingSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FindingSeverity::Info => write!(f, "INFO"),
            FindingSeverity::Low => write!(f, "LOW"),
            FindingSeverity::Medium => write!(f, "MEDIUM"),
            FindingSeverity::High => write!(f, "HIGH"),
            FindingSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Category of defect (aligned with OIP categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
    /// Missing or incorrect trait bounds
    TraitBounds,
    /// Syntax/structure issues, macro expansion bugs
    AstTransform,
    /// Ownership/lifetime errors
    OwnershipBorrow,
    /// Config/environment issues
    ConfigurationErrors,
    /// Race conditions, data races
    ConcurrencyBugs,
    /// Security issues
    SecurityVulnerabilities,
    /// Type mismatches
    TypeErrors,
    /// Memory bugs (use-after-free, etc.)
    MemorySafety,
    /// Logic errors
    LogicErrors,
    /// Performance issues
    PerformanceIssues,
    /// Unknown/uncategorized
    Unknown,
}

impl std::fmt::Display for DefectCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefectCategory::TraitBounds => write!(f, "TraitBounds"),
            DefectCategory::AstTransform => write!(f, "ASTTransform"),
            DefectCategory::OwnershipBorrow => write!(f, "OwnershipBorrow"),
            DefectCategory::ConfigurationErrors => write!(f, "ConfigurationErrors"),
            DefectCategory::ConcurrencyBugs => write!(f, "ConcurrencyBugs"),
            DefectCategory::SecurityVulnerabilities => write!(f, "SecurityVulnerabilities"),
            DefectCategory::TypeErrors => write!(f, "TypeErrors"),
            DefectCategory::MemorySafety => write!(f, "MemorySafety"),
            DefectCategory::LogicErrors => write!(f, "LogicErrors"),
            DefectCategory::PerformanceIssues => write!(f, "PerformanceIssues"),
            DefectCategory::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A single bug finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Unique identifier
    pub id: String,

    /// File path where the finding was located
    pub file: PathBuf,

    /// Line number (1-indexed)
    pub line: usize,

    /// Column number (1-indexed, optional)
    pub column: Option<usize>,

    /// Finding title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: FindingSeverity,

    /// Defect category
    pub category: DefectCategory,

    /// Suspiciousness score (0.0 - 1.0)
    pub suspiciousness: f64,

    /// Hunt mode that discovered this finding
    pub discovered_by: HuntMode,

    /// Evidence supporting the finding
    pub evidence: Vec<FindingEvidence>,

    /// Suggested fix (if available)
    pub suggested_fix: Option<String>,

    /// Related findings (by ID)
    pub related: Vec<String>,
}

impl Finding {
    /// Create a new finding.
    pub fn new(
        id: impl Into<String>,
        file: impl Into<PathBuf>,
        line: usize,
        title: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            file: file.into(),
            line,
            column: None,
            title: title.into(),
            description: String::new(),
            severity: FindingSeverity::Medium,
            category: DefectCategory::Unknown,
            suspiciousness: 0.5,
            discovered_by: HuntMode::Analyze,
            evidence: Vec::new(),
            suggested_fix: None,
            related: Vec::new(),
        }
    }

    /// Set column.
    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: FindingSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set category.
    pub fn with_category(mut self, category: DefectCategory) -> Self {
        self.category = category;
        self
    }

    /// Set suspiciousness score.
    pub fn with_suspiciousness(mut self, score: f64) -> Self {
        self.suspiciousness = score.clamp(0.0, 1.0);
        self
    }

    /// Set discovery mode.
    pub fn with_discovered_by(mut self, mode: HuntMode) -> Self {
        self.discovered_by = mode;
        self
    }

    /// Add evidence.
    pub fn with_evidence(mut self, evidence: FindingEvidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Set suggested fix.
    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Get location string.
    pub fn location(&self) -> String {
        match self.column {
            Some(col) => format!("{}:{}:{}", self.file.display(), self.line, col),
            None => format!("{}:{}", self.file.display(), self.line),
        }
    }
}

/// Evidence supporting a finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingEvidence {
    /// Evidence type
    pub evidence_type: EvidenceKind,

    /// Description
    pub description: String,

    /// Raw data
    pub data: Option<String>,
}

impl FindingEvidence {
    /// Create mutation evidence.
    pub fn mutation(mutant_id: impl Into<String>, survived: bool) -> Self {
        Self {
            evidence_type: EvidenceKind::MutationSurvival,
            description: format!(
                "Mutant {} {}",
                mutant_id.into(),
                if survived { "SURVIVED" } else { "KILLED" }
            ),
            data: Some(if survived {
                "SURVIVED".into()
            } else {
                "KILLED".into()
            }),
        }
    }

    /// Create SBFL evidence.
    pub fn sbfl(formula: impl Into<String>, score: f64) -> Self {
        Self {
            evidence_type: EvidenceKind::SbflScore,
            description: format!("{} suspiciousness: {:.3}", formula.into(), score),
            data: Some(format!("{:.6}", score)),
        }
    }

    /// Create static analysis evidence.
    pub fn static_analysis(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::StaticAnalysis,
            description: format!("[{}] {}", tool.into(), message.into()),
            data: None,
        }
    }

    /// Create fuzzing evidence.
    pub fn fuzzing(input: impl Into<String>, crash_type: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::FuzzingCrash,
            description: crash_type.into(),
            data: Some(input.into()),
        }
    }

    /// Create concolic evidence.
    pub fn concolic(path_constraint: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::ConcolicPath,
            description: "Path constraint solved".into(),
            data: Some(path_constraint.into()),
        }
    }
}

/// Kind of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Mutation survived (test weakness)
    MutationSurvival,
    /// SBFL suspiciousness score
    SbflScore,
    /// Static analysis warning
    StaticAnalysis,
    /// Fuzzing crash/violation
    FuzzingCrash,
    /// Concolic execution path
    ConcolicPath,
    /// LLM classification
    LlmClassification,
    /// Git history correlation
    GitHistory,
}

/// Configuration for a bug hunt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntConfig {
    /// Hunt mode
    pub mode: HuntMode,

    /// Target paths to analyze
    pub targets: Vec<PathBuf>,

    /// Minimum suspiciousness threshold
    pub min_suspiciousness: f64,

    /// Maximum findings to report
    pub max_findings: usize,

    /// SBFL formula (for Hunt/DeepHunt modes)
    pub sbfl_formula: SbflFormula,

    /// Enable LLM filtering
    pub llm_filter: bool,

    /// Fuzzing duration in seconds (for Fuzz mode)
    pub fuzz_duration_secs: u64,

    /// Mutation timeout in seconds (for Falsify mode)
    pub mutation_timeout_secs: u64,

    /// Categories to include (empty = all)
    pub include_categories: Vec<DefectCategory>,

    /// Categories to exclude
    pub exclude_categories: Vec<DefectCategory>,

    // =========================================================================
    // BH-11 to BH-15: Advanced Features
    // =========================================================================

    /// Spec file path (BH-11: Spec-Driven Bug Hunting)
    pub spec_path: Option<PathBuf>,

    /// Spec section filter (e.g., "Authentication")
    pub spec_section: Option<String>,

    /// PMAT ticket reference (BH-12: Ticket Integration)
    pub ticket_ref: Option<String>,

    /// Update spec with findings (BH-14: Bidirectional Linking)
    pub update_spec: bool,

    /// Analyze library only (BH-13: Scoped Analysis)
    pub lib_only: bool,

    /// Analyze specific binary (BH-13: Scoped Analysis)
    pub bin_target: Option<String>,

    /// Exclude test code (BH-13: Scoped Analysis)
    pub exclude_tests: bool,

    /// Suppress known false positive patterns (BH-15)
    pub suppress_false_positives: bool,
}

impl Default for HuntConfig {
    fn default() -> Self {
        Self {
            mode: HuntMode::Analyze,
            targets: vec![PathBuf::from("src")],
            min_suspiciousness: 0.5,
            max_findings: 50,
            sbfl_formula: SbflFormula::Ochiai,
            llm_filter: false,
            fuzz_duration_secs: 60,
            mutation_timeout_secs: 30,
            include_categories: Vec::new(),
            exclude_categories: Vec::new(),
            // BH-11 to BH-15 defaults
            spec_path: None,
            spec_section: None,
            ticket_ref: None,
            update_spec: false,
            lib_only: false,
            bin_target: None,
            exclude_tests: true, // Default to excluding tests
            suppress_false_positives: true, // Default to suppressing
        }
    }
}

/// SBFL formula for fault localization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SbflFormula {
    /// Tarantula formula (classic)
    Tarantula,
    /// Ochiai formula (cosine similarity)
    #[default]
    Ochiai,
    /// DStar with power 2
    DStar2,
    /// DStar with power 3
    DStar3,
}

impl std::fmt::Display for SbflFormula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SbflFormula::Tarantula => write!(f, "Tarantula"),
            SbflFormula::Ochiai => write!(f, "Ochiai"),
            SbflFormula::DStar2 => write!(f, "DStar2"),
            SbflFormula::DStar3 => write!(f, "DStar3"),
        }
    }
}

/// Result of a bug hunt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntResult {
    /// Project path analyzed
    pub project_path: PathBuf,

    /// Hunt mode used
    pub mode: HuntMode,

    /// Configuration used
    pub config: HuntConfig,

    /// Findings discovered
    pub findings: Vec<Finding>,

    /// Statistics
    pub stats: HuntStats,

    /// Timestamp
    pub timestamp: String,

    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl HuntResult {
    /// Create a new hunt result.
    pub fn new(project_path: impl Into<PathBuf>, mode: HuntMode, config: HuntConfig) -> Self {
        Self {
            project_path: project_path.into(),
            mode,
            config,
            findings: Vec::new(),
            stats: HuntStats::default(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_ms: 0,
        }
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: Finding) {
        self.findings.push(finding);
    }

    /// Set duration.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Finalize statistics.
    pub fn finalize(&mut self) {
        self.stats = HuntStats::from_findings(&self.findings);
    }

    /// Get findings sorted by suspiciousness (descending).
    pub fn top_findings(&self, n: usize) -> Vec<&Finding> {
        let mut sorted: Vec<_> = self.findings.iter().collect();
        sorted.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} mode: {} findings ({} critical, {} high) in {}ms",
            self.mode,
            self.findings.len(),
            self.stats.by_severity.get(&FindingSeverity::Critical).unwrap_or(&0),
            self.stats.by_severity.get(&FindingSeverity::High).unwrap_or(&0),
            self.duration_ms
        )
    }
}

/// Statistics from a bug hunt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuntStats {
    /// Total findings
    pub total_findings: usize,

    /// Findings by severity
    pub by_severity: HashMap<FindingSeverity, usize>,

    /// Findings by category
    pub by_category: HashMap<DefectCategory, usize>,

    /// Files analyzed
    pub files_analyzed: usize,

    /// Lines analyzed
    pub lines_analyzed: usize,

    /// Average suspiciousness
    pub avg_suspiciousness: f64,

    /// Max suspiciousness
    pub max_suspiciousness: f64,

    /// Mode-specific stats
    pub mode_stats: ModeStats,
}

impl HuntStats {
    /// Compute statistics from findings.
    pub fn from_findings(findings: &[Finding]) -> Self {
        let mut by_severity: HashMap<FindingSeverity, usize> = HashMap::new();
        let mut by_category: HashMap<DefectCategory, usize> = HashMap::new();
        let mut total_suspiciousness = 0.0;
        let mut max_suspiciousness = 0.0;

        for finding in findings {
            *by_severity.entry(finding.severity).or_default() += 1;
            *by_category.entry(finding.category).or_default() += 1;
            total_suspiciousness += finding.suspiciousness;
            if finding.suspiciousness > max_suspiciousness {
                max_suspiciousness = finding.suspiciousness;
            }
        }

        Self {
            total_findings: findings.len(),
            by_severity,
            by_category,
            avg_suspiciousness: if findings.is_empty() {
                0.0
            } else {
                total_suspiciousness / findings.len() as f64
            },
            max_suspiciousness,
            ..Default::default()
        }
    }
}

/// Mode-specific statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModeStats {
    /// Mutation testing: total mutants
    pub mutants_total: usize,
    /// Mutation testing: killed mutants
    pub mutants_killed: usize,
    /// Mutation testing: survived mutants
    pub mutants_survived: usize,

    /// SBFL: passing tests
    pub sbfl_passing_tests: usize,
    /// SBFL: failing tests
    pub sbfl_failing_tests: usize,

    /// Fuzzing: total executions
    pub fuzz_executions: usize,
    /// Fuzzing: crashes found
    pub fuzz_crashes: usize,
    /// Fuzzing: coverage percentage
    pub fuzz_coverage: f64,

    /// Concolic: paths explored
    pub concolic_paths: usize,
    /// Concolic: constraints solved
    pub concolic_constraints_solved: usize,
    /// Concolic: timeouts
    pub concolic_timeouts: usize,

    /// LLM: warnings filtered
    pub llm_filtered: usize,
    /// LLM: true positives retained
    pub llm_retained: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BH-TYP-001: HuntMode Display
    // =========================================================================

    #[test]
    fn test_bh_typ_001_hunt_mode_display() {
        assert_eq!(format!("{}", HuntMode::Falsify), "Falsify");
        assert_eq!(format!("{}", HuntMode::Hunt), "Hunt");
        assert_eq!(format!("{}", HuntMode::Analyze), "Analyze");
        assert_eq!(format!("{}", HuntMode::Fuzz), "Fuzz");
        assert_eq!(format!("{}", HuntMode::DeepHunt), "DeepHunt");
    }

    // =========================================================================
    // BH-TYP-002: FindingSeverity Ordering
    // =========================================================================

    #[test]
    fn test_bh_typ_002_finding_severity_ordering() {
        assert!(FindingSeverity::Critical > FindingSeverity::High);
        assert!(FindingSeverity::High > FindingSeverity::Medium);
        assert!(FindingSeverity::Medium > FindingSeverity::Low);
        assert!(FindingSeverity::Low > FindingSeverity::Info);
    }

    #[test]
    fn test_bh_typ_002_finding_severity_display() {
        assert_eq!(format!("{}", FindingSeverity::Critical), "CRITICAL");
        assert_eq!(format!("{}", FindingSeverity::High), "HIGH");
        assert_eq!(format!("{}", FindingSeverity::Medium), "MEDIUM");
        assert_eq!(format!("{}", FindingSeverity::Low), "LOW");
        assert_eq!(format!("{}", FindingSeverity::Info), "INFO");
    }

    // =========================================================================
    // BH-TYP-003: Finding Builder
    // =========================================================================

    #[test]
    fn test_bh_typ_003_finding_builder() {
        let finding = Finding::new("BH-001", "src/lib.rs", 42, "Potential null dereference")
            .with_column(10)
            .with_description("This line may dereference a null pointer")
            .with_severity(FindingSeverity::High)
            .with_category(DefectCategory::MemorySafety)
            .with_suspiciousness(0.95);

        assert_eq!(finding.id, "BH-001");
        assert_eq!(finding.line, 42);
        assert_eq!(finding.column, Some(10));
        assert_eq!(finding.severity, FindingSeverity::High);
        assert_eq!(finding.category, DefectCategory::MemorySafety);
        assert!((finding.suspiciousness - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_bh_typ_003_finding_location() {
        let finding = Finding::new("BH-001", "src/lib.rs", 42, "Test");
        assert_eq!(finding.location(), "src/lib.rs:42");

        let finding_with_col = finding.with_column(10);
        assert_eq!(finding_with_col.location(), "src/lib.rs:42:10");
    }

    // =========================================================================
    // BH-TYP-004: Evidence Types
    // =========================================================================

    #[test]
    fn test_bh_typ_004_evidence_mutation() {
        let evidence = FindingEvidence::mutation("mut_001", true);
        assert_eq!(evidence.evidence_type, EvidenceKind::MutationSurvival);
        assert!(evidence.description.contains("SURVIVED"));
    }

    #[test]
    fn test_bh_typ_004_evidence_sbfl() {
        let evidence = FindingEvidence::sbfl("Ochiai", 0.875);
        assert_eq!(evidence.evidence_type, EvidenceKind::SbflScore);
        assert!(evidence.description.contains("0.875"));
    }

    #[test]
    fn test_bh_typ_004_evidence_static() {
        let evidence = FindingEvidence::static_analysis("clippy", "unused variable");
        assert_eq!(evidence.evidence_type, EvidenceKind::StaticAnalysis);
        assert!(evidence.description.contains("[clippy]"));
    }

    // =========================================================================
    // BH-TYP-005: HuntConfig Defaults
    // =========================================================================

    #[test]
    fn test_bh_typ_005_hunt_config_defaults() {
        let config = HuntConfig::default();
        assert_eq!(config.mode, HuntMode::Analyze);
        assert_eq!(config.max_findings, 50);
        assert!((config.min_suspiciousness - 0.5).abs() < 0.001);
        assert_eq!(config.sbfl_formula, SbflFormula::Ochiai);
        assert!(!config.llm_filter);
    }

    // =========================================================================
    // BH-TYP-006: HuntResult Operations
    // =========================================================================

    #[test]
    fn test_bh_typ_006_hunt_result_new() {
        let config = HuntConfig::default();
        let result = HuntResult::new(".", HuntMode::Analyze, config);

        assert_eq!(result.mode, HuntMode::Analyze);
        assert!(result.findings.is_empty());
    }

    #[test]
    fn test_bh_typ_006_hunt_result_add_finding() {
        let config = HuntConfig::default();
        let mut result = HuntResult::new(".", HuntMode::Analyze, config);

        result.add_finding(Finding::new("F-001", "test.rs", 1, "Test finding"));
        result.add_finding(Finding::new("F-002", "test.rs", 2, "Another finding"));

        assert_eq!(result.findings.len(), 2);
    }

    #[test]
    fn test_bh_typ_006_hunt_result_top_findings() {
        let config = HuntConfig::default();
        let mut result = HuntResult::new(".", HuntMode::Analyze, config);

        result.add_finding(
            Finding::new("F-001", "test.rs", 1, "Low")
                .with_suspiciousness(0.3),
        );
        result.add_finding(
            Finding::new("F-002", "test.rs", 2, "High")
                .with_suspiciousness(0.9),
        );
        result.add_finding(
            Finding::new("F-003", "test.rs", 3, "Medium")
                .with_suspiciousness(0.6),
        );

        let top = result.top_findings(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, "F-002"); // Highest suspiciousness
        assert_eq!(top[1].id, "F-003");
    }

    // =========================================================================
    // BH-TYP-007: HuntStats Computation
    // =========================================================================

    #[test]
    fn test_bh_typ_007_hunt_stats_from_findings() {
        let findings = vec![
            Finding::new("F-001", "test.rs", 1, "Critical")
                .with_severity(FindingSeverity::Critical)
                .with_category(DefectCategory::MemorySafety)
                .with_suspiciousness(0.9),
            Finding::new("F-002", "test.rs", 2, "High")
                .with_severity(FindingSeverity::High)
                .with_category(DefectCategory::ConcurrencyBugs)
                .with_suspiciousness(0.7),
            Finding::new("F-003", "test.rs", 3, "Medium")
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::MemorySafety)
                .with_suspiciousness(0.5),
        ];

        let stats = HuntStats::from_findings(&findings);

        assert_eq!(stats.total_findings, 3);
        assert_eq!(stats.by_severity.get(&FindingSeverity::Critical), Some(&1));
        assert_eq!(stats.by_severity.get(&FindingSeverity::High), Some(&1));
        assert_eq!(stats.by_severity.get(&FindingSeverity::Medium), Some(&1));
        assert_eq!(stats.by_category.get(&DefectCategory::MemorySafety), Some(&2));
        assert!((stats.avg_suspiciousness - 0.7).abs() < 0.001);
        assert!((stats.max_suspiciousness - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_bh_typ_007_hunt_stats_empty() {
        let stats = HuntStats::from_findings(&[]);
        assert_eq!(stats.total_findings, 0);
        assert_eq!(stats.avg_suspiciousness, 0.0);
    }

    // =========================================================================
    // BH-TYP-008: SbflFormula Display
    // =========================================================================

    #[test]
    fn test_bh_typ_008_sbfl_formula_display() {
        assert_eq!(format!("{}", SbflFormula::Tarantula), "Tarantula");
        assert_eq!(format!("{}", SbflFormula::Ochiai), "Ochiai");
        assert_eq!(format!("{}", SbflFormula::DStar2), "DStar2");
        assert_eq!(format!("{}", SbflFormula::DStar3), "DStar3");
    }

    // =========================================================================
    // BH-TYP-009: DefectCategory Display
    // =========================================================================

    #[test]
    fn test_bh_typ_009_defect_category_display() {
        assert_eq!(format!("{}", DefectCategory::MemorySafety), "MemorySafety");
        assert_eq!(format!("{}", DefectCategory::ConcurrencyBugs), "ConcurrencyBugs");
        assert_eq!(format!("{}", DefectCategory::SecurityVulnerabilities), "SecurityVulnerabilities");
    }

    // =========================================================================
    // BH-TYP-010: Suspiciousness Clamping
    // =========================================================================

    #[test]
    fn test_bh_typ_010_suspiciousness_clamping() {
        let finding = Finding::new("F-001", "test.rs", 1, "Test")
            .with_suspiciousness(1.5); // Above 1.0
        assert!((finding.suspiciousness - 1.0).abs() < 0.001);

        let finding = Finding::new("F-002", "test.rs", 1, "Test")
            .with_suspiciousness(-0.5); // Below 0.0
        assert!((finding.suspiciousness - 0.0).abs() < 0.001);
    }
}
