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
