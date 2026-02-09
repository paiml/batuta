use super::*;
use crate::bug_hunter::{Finding, FindingSeverity};

#[test]
fn test_parse_claim_header_standard() {
    let result = parse_claim_header("### BH-01: Mutation-Based Invariant Falsification");
    assert!(result.is_some());
    let (id, title) = result.unwrap();
    assert_eq!(id, "BH-01");
    assert_eq!(title, "Mutation-Based Invariant Falsification");
}

#[test]
fn test_parse_claim_header_auth() {
    let result = parse_claim_header("### AUTH-01: Token Validation");
    assert!(result.is_some());
    let (id, title) = result.unwrap();
    assert_eq!(id, "AUTH-01");
    assert_eq!(title, "Token Validation");
}

#[test]
fn test_parse_claim_header_no_id() {
    let result = parse_claim_header("### Just a Title");
    assert!(result.is_none());
}

#[test]
fn test_parse_claims_from_content() {
    let content = r#"
## Section 1: Test

### BH-01: First Claim

Some text here.

### BH-02: Second Claim

More text.
"#;
    let claims = parse_claims(content);
    assert_eq!(claims.len(), 2);
    assert_eq!(claims[0].id, "BH-01");
    assert_eq!(claims[1].id, "BH-02");
}

#[test]
fn test_claim_status_display() {
    assert_eq!(format!("{}", ClaimStatus::Verified), "✓ Verified");
    assert_eq!(format!("{}", ClaimStatus::Warning), "⚠️ Warning");
    assert_eq!(format!("{}", ClaimStatus::Failed), "✗ Failed");
    assert_eq!(format!("{}", ClaimStatus::Pending), "○ Pending");
}

#[test]
fn test_remove_existing_status_blocks() {
    let content = r#"### BH-01: Test Claim

<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->

Some content here.
"#;
    let result = remove_existing_status_blocks(content);
    assert!(!result.contains("bug-hunter-status"));
    assert!(result.contains("### BH-01: Test Claim"));
    assert!(result.contains("Some content here."));
}

#[test]
fn test_remove_existing_status_blocks_no_block() {
    let content = "### BH-01: Test\n\nNo status block here.\n";
    let result = remove_existing_status_blocks(content);
    assert_eq!(result, "### BH-01: Test\n\nNo status block here.\n");
}

#[test]
fn test_find_claim_end() {
    let content = "### BH-01: First Claim\n\nSome text.\n\n### BH-02: Second\n";
    let pos = find_claim_end(content, "BH-01");
    assert!(pos.is_some());
    assert!(pos.unwrap() > 0);
}

#[test]
fn test_find_claim_end_not_found() {
    let content = "### BH-01: First Claim\n";
    let pos = find_claim_end(content, "BH-99");
    assert!(pos.is_none());
}

#[test]
fn test_code_location_creation() {
    let loc = CodeLocation {
        file: PathBuf::from("src/main.rs"),
        line: 42,
        context: "fn main()".to_string(),
    };
    assert_eq!(loc.file, PathBuf::from("src/main.rs"));
    assert_eq!(loc.line, 42);
    assert_eq!(loc.context, "fn main()");
}

#[test]
fn test_spec_claim_creation() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test Claim".to_string(),
        line: 10,
        section_path: vec!["Section 1".to_string()],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    };
    assert_eq!(claim.id, "TEST-01");
    assert_eq!(claim.status, ClaimStatus::Pending);
}

#[test]
fn test_parse_claim_header_short_prefix() {
    let result = parse_claim_header("### X-1: Short");
    assert!(result.is_some());
    let (id, _) = result.unwrap();
    assert_eq!(id, "X-1");
}

#[test]
fn test_parse_claim_header_long_prefix() {
    let result = parse_claim_header("### ABCD-1234: Long");
    assert!(result.is_some());
    let (id, _) = result.unwrap();
    assert_eq!(id, "ABCD-1234");
}

#[test]
fn test_parse_claim_header_too_long_prefix() {
    // 5 letters is too long
    let result = parse_claim_header("### ABCDE-01: Too Long");
    assert!(result.is_none());
}

#[test]
fn test_parse_claim_header_lowercase() {
    // Lowercase not allowed
    let result = parse_claim_header("### abc-01: Lower");
    assert!(result.is_none());
}

#[test]
fn test_parse_claim_header_no_colon() {
    let result = parse_claim_header("### BH-01 No Colon");
    assert!(result.is_none());
}

#[test]
fn test_parse_claims_with_sections() {
    let content = r#"
## Section One

### CB-001: First

## Section Two

### CB-002: Second
"#;
    let claims = parse_claims(content);
    assert_eq!(claims.len(), 2);
    assert!(claims[0].section_path.contains(&"Section One".to_string()));
    assert!(claims[1].section_path.contains(&"Section Two".to_string()));
}

#[test]
fn test_parse_claims_with_subsections() {
    let content = r#"
## Main Section

### Sub Section

#### CB-001: Claim
"#;
    let claims = parse_claims(content);
    // ### is parsed as claim header if it matches the pattern
    // But "Sub Section" doesn't match claim ID pattern
    assert_eq!(claims.len(), 0); // "CB-001" is under ####, not ###
}

#[test]
fn test_parsed_spec_claims_for_section() {
    let spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![
            SpecClaim {
                id: "BH-01".to_string(),
                title: "Bug Hunt".to_string(),
                line: 1,
                section_path: vec!["Bug Hunting".to_string()],
                implementations: vec![],
                findings: vec![],
                status: ClaimStatus::Pending,
            },
            SpecClaim {
                id: "AUTH-01".to_string(),
                title: "Auth Check".to_string(),
                line: 10,
                section_path: vec!["Authentication".to_string()],
                implementations: vec![],
                findings: vec![],
                status: ClaimStatus::Pending,
            },
        ],
        original_content: String::new(),
    };

    let bh_claims = spec.claims_for_section("Bug");
    assert_eq!(bh_claims.len(), 1);
    assert_eq!(bh_claims[0].id, "BH-01");

    let auth_claims = spec.claims_for_section("AUTH");
    assert_eq!(auth_claims.len(), 1);
    assert_eq!(auth_claims[0].id, "AUTH-01");
}

#[test]
fn test_claim_status_equality() {
    assert_eq!(ClaimStatus::Verified, ClaimStatus::Verified);
    assert_ne!(ClaimStatus::Verified, ClaimStatus::Failed);
}

// =========================================================================
// BH-SPEC-009: Additional Coverage Tests
// =========================================================================

#[test]
fn test_generate_status_block_verified() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Verified,
    };
    let findings: Vec<Finding> = vec![];
    let block = generate_status_block(&claim, &findings);
    assert!(block.contains("bug-hunter-status"));
    assert!(block.contains("Verified"));
    assert!(block.contains("None ✓"));
}

#[test]
fn test_generate_status_block_with_findings() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Warning,
    };
    let findings = vec![
        Finding::new("F-001", "test.rs", 10, "Test finding")
            .with_severity(FindingSeverity::Low),
    ];
    let block = generate_status_block(&claim, &findings);
    assert!(block.contains("1 issue(s)"));
    assert!(block.contains("F-001"));
}

#[test]
fn test_generate_status_block_with_implementations() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![CodeLocation {
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            context: "fn impl_func()".to_string(),
        }],
        findings: vec![],
        status: ClaimStatus::Verified,
    };
    let findings: Vec<Finding> = vec![];
    let block = generate_status_block(&claim, &findings);
    assert!(block.contains("Implementations:"));
    assert!(block.contains("src/lib.rs:42"));
}

#[test]
fn test_generate_status_block_many_findings() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Failed,
    };
    // Create more than 5 findings
    let findings: Vec<Finding> = (0..10)
        .map(|i| {
            Finding::new(format!("F-{:03}", i), "test.rs", i, format!("Finding {}", i))
        })
        .collect();
    let block = generate_status_block(&claim, &findings);
    assert!(block.contains("10 issue(s)"));
    assert!(block.contains("and 5 more"));
}

#[test]
fn test_update_with_findings_verified() {
    let mut spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![SpecClaim {
            id: "BH-01".to_string(),
            title: "Test Claim".to_string(),
            line: 1,
            section_path: vec![],
            implementations: vec![],
            findings: vec![],
            status: ClaimStatus::Pending,
        }],
        original_content: "### BH-01: Test Claim\n\nSome content.\n".to_string(),
    };

    let findings: Vec<(String, Vec<Finding>)> = vec![("BH-01".to_string(), vec![])];
    let result = spec.update_with_findings(&findings);
    assert!(result.is_ok());
    let updated = result.unwrap();
    assert!(updated.contains("Verified"));
    assert_eq!(spec.claims[0].status, ClaimStatus::Verified);
}

#[test]
fn test_update_with_findings_warning() {
    let mut spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![SpecClaim {
            id: "BH-01".to_string(),
            title: "Test Claim".to_string(),
            line: 1,
            section_path: vec![],
            implementations: vec![],
            findings: vec![],
            status: ClaimStatus::Pending,
        }],
        original_content: "### BH-01: Test Claim\n\nSome content.\n".to_string(),
    };

    let findings: Vec<(String, Vec<Finding>)> = vec![(
        "BH-01".to_string(),
        vec![Finding::new("F-001", "test.rs", 1, "Low severity")
            .with_severity(FindingSeverity::Low)],
    )];
    let result = spec.update_with_findings(&findings);
    assert!(result.is_ok());
    assert_eq!(spec.claims[0].status, ClaimStatus::Warning);
}

#[test]
fn test_update_with_findings_failed() {
    let mut spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![SpecClaim {
            id: "BH-01".to_string(),
            title: "Test Claim".to_string(),
            line: 1,
            section_path: vec![],
            implementations: vec![],
            findings: vec![],
            status: ClaimStatus::Pending,
        }],
        original_content: "### BH-01: Test Claim\n\nSome content.\n".to_string(),
    };

    let findings: Vec<(String, Vec<Finding>)> = vec![(
        "BH-01".to_string(),
        vec![Finding::new("F-001", "test.rs", 1, "Critical issue")
            .with_severity(FindingSeverity::Critical)],
    )];
    let result = spec.update_with_findings(&findings);
    assert!(result.is_ok());
    assert_eq!(spec.claims[0].status, ClaimStatus::Failed);
}

#[test]
fn test_update_with_findings_unknown_claim() {
    let mut spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![SpecClaim {
            id: "BH-01".to_string(),
            title: "Test Claim".to_string(),
            line: 1,
            section_path: vec![],
            implementations: vec![],
            findings: vec![],
            status: ClaimStatus::Pending,
        }],
        original_content: "### BH-01: Test Claim\n\nSome content.\n".to_string(),
    };

    // Findings for a claim that doesn't exist
    let findings: Vec<(String, Vec<Finding>)> = vec![("BH-99".to_string(), vec![])];
    let result = spec.update_with_findings(&findings);
    assert!(result.is_ok());
    // Original claim should still be pending
    assert_eq!(spec.claims[0].status, ClaimStatus::Pending);
}

#[test]
fn test_parsed_spec_parse_nonexistent_file() {
    let result = ParsedSpec::parse(Path::new("/nonexistent/file.md"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

#[test]
fn test_code_location_serialization() {
    let loc = CodeLocation {
        file: PathBuf::from("src/main.rs"),
        line: 42,
        context: "fn main()".to_string(),
    };
    let json = serde_json::to_string(&loc).unwrap();
    let deserialized: CodeLocation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.file, loc.file);
    assert_eq!(deserialized.line, loc.line);
}

#[test]
fn test_spec_claim_serialization() {
    let claim = SpecClaim {
        id: "TEST-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec!["Section".to_string()],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Verified,
    };
    let json = serde_json::to_string(&claim).unwrap();
    let deserialized: SpecClaim = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, claim.id);
    assert_eq!(deserialized.status, ClaimStatus::Verified);
}

#[test]
fn test_parsed_spec_serialization() {
    let spec = ParsedSpec {
        path: PathBuf::from("test.md"),
        claims: vec![],
        original_content: "# Test".to_string(),
    };
    let json = serde_json::to_string(&spec).unwrap();
    let deserialized: ParsedSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.path, spec.path);
}

// =========================================================================
// Coverage gap: map_findings_to_claims
// =========================================================================

#[test]
fn test_map_findings_to_claims_basic() {
    let temp = std::env::temp_dir().join("test_spec_map");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // Write a source file that references our claim ID
    std::fs::write(
        temp.join("src/lib.rs"),
        "// Implements BH-01: mutation testing\nfn run_mutations() {}\n",
    )
    .unwrap();

    let claims = vec![SpecClaim {
        id: "BH-01".to_string(),
        title: "Mutation Testing".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    }];

    let findings = vec![Finding::new("F-001", temp.join("src/lib.rs"), 2, "Pattern: unwrap")
        .with_severity(FindingSeverity::Medium)];

    let mapping = map_findings_to_claims(&claims, &findings, &temp);

    // Should have entry for BH-01
    assert!(mapping.contains_key("BH-01"));
    // Finding at line 2 is within 50 lines of implementation at line 1
    let claim_findings = &mapping["BH-01"];
    assert_eq!(
        claim_findings.len(),
        1,
        "Finding near implementation should be mapped"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_map_findings_to_claims_no_match() {
    let temp = std::env::temp_dir().join("test_spec_map_nomatch");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // Source file with claim reference
    std::fs::write(
        temp.join("src/lib.rs"),
        "// BH-01 implemented here\nfn impl_bh01() {}\n",
    )
    .unwrap();

    let claims = vec![SpecClaim {
        id: "BH-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    }];

    // Finding in a completely different file
    let findings = vec![Finding::new("F-001", PathBuf::from("src/other.rs"), 100, "Pattern: TODO")
        .with_severity(FindingSeverity::Low)];

    let mapping = map_findings_to_claims(&claims, &findings, &temp);
    let claim_findings = &mapping["BH-01"];
    assert!(
        claim_findings.is_empty(),
        "Finding in different file should not be mapped"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_map_findings_to_claims_far_line() {
    let temp = std::env::temp_dir().join("test_spec_map_far");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    // Source file with claim reference at line 1
    let mut content = "// BH-01 implemented here\n".to_string();
    for i in 0..100 {
        content.push_str(&format!("fn line_{}() {{}}\n", i));
    }
    std::fs::write(temp.join("src/lib.rs"), &content).unwrap();

    let claims = vec![SpecClaim {
        id: "BH-01".to_string(),
        title: "Test".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    }];

    // Finding at line 90 — more than 50 lines away from implementation at line 1
    let findings = vec![
        Finding::new("F-001", temp.join("src/lib.rs"), 90, "Pattern: HACK")
            .with_severity(FindingSeverity::Medium),
    ];

    let mapping = map_findings_to_claims(&claims, &findings, &temp);
    let claim_findings = &mapping["BH-01"];
    assert!(
        claim_findings.is_empty(),
        "Finding >50 lines from implementation should not be mapped"
    );

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// Coverage gap: find_implementations
// =========================================================================

#[test]
fn test_find_implementations_basic() {
    let temp = std::env::temp_dir().join("test_spec_find_impl");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(
        temp.join("src/lib.rs"),
        "/// Implements AUTH-01 token validation\nfn validate_token() {}\n",
    )
    .unwrap();

    let claim = SpecClaim {
        id: "AUTH-01".to_string(),
        title: "Token Validation".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    };

    let impls = find_implementations(&claim, &temp);
    assert!(
        !impls.is_empty(),
        "Should find implementation referencing AUTH-01"
    );
    assert_eq!(impls[0].line, 1);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_find_implementations_no_match() {
    let temp = std::env::temp_dir().join("test_spec_find_impl_none");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(temp.join("src"));

    std::fs::write(temp.join("src/lib.rs"), "fn main() {}\n").unwrap();

    let claim = SpecClaim {
        id: "NONEXIST-99".to_string(),
        title: "Missing".to_string(),
        line: 1,
        section_path: vec![],
        implementations: vec![],
        findings: vec![],
        status: ClaimStatus::Pending,
    };

    let impls = find_implementations(&claim, &temp);
    assert!(impls.is_empty());

    let _ = std::fs::remove_dir_all(&temp);
}
