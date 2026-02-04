//! Extended tests for publish status module (PUB-009 through PUB-014).

use std::path::{Path, PathBuf};

use super::cache::{compute_cache_key, PublishStatusCache};
use super::format::format_report_text;
use super::git::{determine_action, get_local_version};
use super::scanner::PublishStatusScanner;
use super::types::{CrateStatus, GitStatus, PublishAction, PublishStatusReport};

// ========================================================================
// PUB-009: Cache key computation tests
// ========================================================================

#[test]
fn test_pub_009_compute_cache_key() {
    let temp_dir = std::env::temp_dir().join("batuta_cache_key_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(
        &cargo_toml,
        "[package]\nname = \"test\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();

    // Initialize git repo
    let _ = std::process::Command::new("git")
        .args(["init"])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let key = compute_cache_key(&temp_dir);
    assert!(key.is_ok());
    // Key should be a hex string
    assert!(key.unwrap().chars().all(|c| c.is_ascii_hexdigit()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_009_compute_cache_key_no_cargo_toml() {
    let result = compute_cache_key(Path::new("/nonexistent/path"));
    assert!(result.is_err());
}

#[test]
fn test_pub_009_compute_cache_key_no_git() {
    let temp_dir = std::env::temp_dir().join("batuta_no_git_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(
        &cargo_toml,
        "[package]\nname = \"test\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();

    // Should still work, using "no-git" as HEAD
    let key = compute_cache_key(&temp_dir);
    assert!(key.is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-010: Scanner tests
// ========================================================================

#[test]
fn test_pub_010_scanner_find_crate_dirs_empty() {
    let temp_dir = std::env::temp_dir().join("batuta_scanner_empty_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let scanner = PublishStatusScanner::new(temp_dir.clone());
    let dirs = scanner.find_crate_dirs();
    assert!(dirs.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_010_scanner_new() {
    let scanner = PublishStatusScanner::new(PathBuf::from("/tmp"));
    assert_eq!(scanner.workspace_root, PathBuf::from("/tmp"));
}

// ========================================================================
// PUB-011: GitStatus additional tests
// ========================================================================

#[test]
fn test_pub_011_git_status_summary_untracked_only() {
    let status = GitStatus {
        modified: 0,
        untracked: 3,
        staged: 0,
        head_sha: String::new(),
        is_clean: false,
    };
    assert_eq!(status.summary(), "3?");
}

#[test]
fn test_pub_011_git_status_summary_staged_only() {
    let status = GitStatus {
        modified: 0,
        untracked: 0,
        staged: 2,
        head_sha: String::new(),
        is_clean: false,
    };
    assert_eq!(status.summary(), "2+");
}

#[test]
fn test_pub_011_git_status_default() {
    let status = GitStatus::default();
    // Default has is_clean = false (since the Default derive sets it to false)
    assert_eq!(status.total_changes(), 0);
    assert_eq!(status.modified, 0);
    assert_eq!(status.untracked, 0);
    assert_eq!(status.staged, 0);
}

// ========================================================================
// PUB-012: CrateStatus tests
// ========================================================================

#[test]
fn test_pub_012_crate_status_with_error() {
    let status = CrateStatus {
        name: "broken".to_string(),
        local_version: None,
        crates_io_version: None,
        git_status: GitStatus::default(),
        action: PublishAction::Error,
        path: PathBuf::from("/broken"),
        error: Some("Test error".to_string()),
    };

    assert_eq!(status.action, PublishAction::Error);
    assert!(status.error.is_some());
}

#[test]
fn test_pub_012_crate_status_serialization() {
    let status = CrateStatus {
        name: "test".to_string(),
        local_version: Some("1.0.0".to_string()),
        crates_io_version: Some("1.0.0".to_string()),
        git_status: GitStatus::default(),
        action: PublishAction::UpToDate,
        path: PathBuf::from("."),
        error: None,
    };

    let json = serde_json::to_string(&status).unwrap();
    assert!(json.contains("\"name\":\"test\""));

    let deserialized: CrateStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "test");
}

// ========================================================================
// PUB-013: Report edge cases
// ========================================================================

#[test]
fn test_pub_013_report_empty() {
    let report = PublishStatusReport::from_statuses(vec![], 0, 0);
    assert_eq!(report.total, 0);
    assert_eq!(report.needs_publish, 0);
    assert_eq!(report.needs_commit, 0);
    assert_eq!(report.up_to_date, 0);
}

#[test]
fn test_pub_013_report_all_errors() {
    let statuses = vec![
        CrateStatus {
            name: "a".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::from("."),
            error: Some("error 1".to_string()),
        },
        CrateStatus {
            name: "b".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::from("."),
            error: Some("error 2".to_string()),
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 0, 0);
    assert_eq!(report.total, 2);
    assert_eq!(report.needs_publish, 0);
    assert_eq!(report.up_to_date, 0);
}

#[test]
fn test_pub_013_format_report_text_with_missing() {
    let statuses = vec![CrateStatus {
        name: "missing".to_string(),
        local_version: None,
        crates_io_version: None,
        git_status: GitStatus::default(),
        action: PublishAction::Error,
        path: PathBuf::from("."),
        error: None,
    }];

    let report = PublishStatusReport::from_statuses(statuses, 0, 0);
    let text = format_report_text(&report);

    assert!(text.contains("missing"));
    assert!(text.contains("-")); // Missing versions shown as "-"
}

// ========================================================================
// PUB-014: Action edge cases
// ========================================================================

#[test]
fn test_pub_014_determine_action_not_published_dirty() {
    let git = GitStatus {
        is_clean: false,
        modified: 1,
        ..Default::default()
    };
    let action = determine_action(Some("1.0.0"), None, &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_014_determine_action_same_version_dirty() {
    let git = GitStatus {
        is_clean: false,
        staged: 1,
        ..Default::default()
    };
    let action = determine_action(Some("1.0.0"), Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_014_determine_action_invalid_semver() {
    let git = GitStatus {
        is_clean: true,
        ..Default::default()
    };
    // Invalid semver versions
    let action = determine_action(Some("not-a-version"), Some("also-not-valid"), &git);
    assert_eq!(action, PublishAction::UpToDate); // Falls through to UpToDate
}
