//! Publish Status Scanner with O(1) Cache
//!
//! Scans PAIML stack repositories for publish status with intelligent caching.
//! Uses content-addressable cache keys to achieve O(1) lookups for unchanged repos.
//!
//! ## Cache Invalidation
//!
//! Cache keys are computed from:
//! - Cargo.toml content hash (blake3)
//! - Git HEAD commit SHA
//! - Worktree modification time
//!
//! crates.io versions are cached with 15-minute TTL.
//!
//! ## Performance Target
//!
//! - Cold cache: <5s (parallel fetches)
//! - Warm cache: <100ms (O(1) hash checks)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::PAIML_CRATES;

// ============================================================================
// PUB-001: Core Types
// ============================================================================

/// Recommended action for a crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublishAction {
    /// Already published and up to date
    UpToDate,
    /// Has uncommitted changes, needs commit first
    NeedsCommit,
    /// Committed but not published
    NeedsPublish,
    /// Local version behind crates.io (unusual)
    LocalBehind,
    /// Not yet published to crates.io
    NotPublished,
    /// Error checking status
    Error,
}

#[allow(dead_code)] // Used by examples and external consumers
impl PublishAction {
    /// Get display symbol for action
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::UpToDate => "✓",
            Self::NeedsCommit => "📝",
            Self::NeedsPublish => "📦",
            Self::LocalBehind => "⚠️",
            Self::NotPublished => "🆕",
            Self::Error => "❌",
        }
    }

    /// Get action description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::UpToDate => "up to date",
            Self::NeedsCommit => "commit changes",
            Self::NeedsPublish => "PUBLISH",
            Self::LocalBehind => "local behind",
            Self::NotPublished => "not published",
            Self::Error => "error",
        }
    }
}

/// Git status summary for a repo
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatus {
    /// Number of modified files
    pub modified: usize,
    /// Number of untracked files
    pub untracked: usize,
    /// Number of staged files
    pub staged: usize,
    /// Current HEAD commit SHA (short)
    pub head_sha: String,
    /// Is repo clean?
    pub is_clean: bool,
}

impl GitStatus {
    /// Total changed files
    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.modified + self.untracked + self.staged
    }

    /// Summary string
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_clean {
            "clean".to_string()
        } else {
            let mut parts = Vec::new();
            if self.modified > 0 {
                parts.push(format!("{}M", self.modified));
            }
            if self.untracked > 0 {
                parts.push(format!("{}?", self.untracked));
            }
            if self.staged > 0 {
                parts.push(format!("{}+", self.staged));
            }
            parts.join(" ")
        }
    }
}

/// Status of a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateStatus {
    /// Crate name
    pub name: String,
    /// Local version from Cargo.toml
    pub local_version: Option<String>,
    /// Published version on crates.io
    pub crates_io_version: Option<String>,
    /// Git status
    pub git_status: GitStatus,
    /// Recommended action
    pub action: PublishAction,
    /// Path to repo
    pub path: PathBuf,
    /// Error message if any
    pub error: Option<String>,
}

/// Cache entry for a single repo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key (hash of Cargo.toml + git HEAD)
    pub cache_key: String,
    /// Cached status
    pub status: CrateStatus,
    /// When crates.io was last checked (Unix timestamp)
    pub crates_io_checked_at: u64,
    /// When this entry was created
    pub created_at: u64,
}

impl CacheEntry {
    /// Check if crates.io data is stale (>15 min)
    #[must_use]
    pub fn is_crates_io_stale(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now - self.crates_io_checked_at > 15 * 60
    }
}

/// Full publish status report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishStatusReport {
    /// Status for each crate
    pub crates: Vec<CrateStatus>,
    /// Total crates checked
    pub total: usize,
    /// Crates needing publish
    pub needs_publish: usize,
    /// Crates needing commit
    pub needs_commit: usize,
    /// Crates up to date
    pub up_to_date: usize,
    /// Cache hits (O(1) lookups)
    pub cache_hits: usize,
    /// Cache misses (required refresh)
    pub cache_misses: usize,
    /// Time to generate report (ms)
    pub elapsed_ms: u64,
}

impl PublishStatusReport {
    /// Create report from crate statuses
    #[must_use]
    pub fn from_statuses(crates: Vec<CrateStatus>, cache_hits: usize, elapsed_ms: u64) -> Self {
        let total = crates.len();
        let needs_publish = crates
            .iter()
            .filter(|c| c.action == PublishAction::NeedsPublish)
            .count();
        let needs_commit = crates
            .iter()
            .filter(|c| c.action == PublishAction::NeedsCommit)
            .count();
        let up_to_date = crates
            .iter()
            .filter(|c| c.action == PublishAction::UpToDate)
            .count();
        let cache_misses = total - cache_hits;

        Self {
            crates,
            total,
            needs_publish,
            needs_commit,
            up_to_date,
            cache_hits,
            cache_misses,
            elapsed_ms,
        }
    }
}

// ============================================================================
// PUB-002: Cache Implementation
// ============================================================================

/// Persistent cache for publish status
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PublishStatusCache {
    /// Cache entries by crate name
    entries: HashMap<String, CacheEntry>,
    /// Cache file path
    #[serde(skip)]
    cache_path: Option<PathBuf>,
}

impl PublishStatusCache {
    /// Default cache path
    fn default_cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("batuta")
            .join("publish-status.json")
    }

    /// Load cache from disk
    #[must_use]
    pub fn load() -> Self {
        let path = Self::default_cache_path();
        Self::load_from(&path).unwrap_or_default()
    }

    /// Load cache from specific path
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(path)?;
        let mut cache: Self = serde_json::from_str(&data)?;
        cache.cache_path = Some(path.to_path_buf());
        Ok(cache)
    }

    /// Save cache to disk
    pub fn save(&self) -> Result<()> {
        let path = self
            .cache_path
            .clone()
            .unwrap_or_else(Self::default_cache_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    /// Get cached entry if valid
    #[must_use]
    pub fn get(&self, name: &str, cache_key: &str) -> Option<&CacheEntry> {
        self.entries.get(name).filter(|e| e.cache_key == cache_key)
    }

    /// Insert or update entry
    pub fn insert(&mut self, name: String, entry: CacheEntry) {
        self.entries.insert(name, entry);
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ============================================================================
// PUB-003: Cache Key Computation
// ============================================================================

/// Compute cache key for a repo
/// Key = blake3(Cargo.toml content || git HEAD SHA || Cargo.toml mtime)
pub fn compute_cache_key(repo_path: &Path) -> Result<String> {
    let cargo_toml = repo_path.join("Cargo.toml");

    if !cargo_toml.exists() {
        return Err(anyhow!("No Cargo.toml found at {:?}", repo_path));
    }

    // Read Cargo.toml content
    let content = std::fs::read(&cargo_toml)?;

    // Get mtime
    let mtime = std::fs::metadata(&cargo_toml)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Get git HEAD (if available)
    let head_sha = get_git_head(repo_path).unwrap_or_else(|_| "no-git".to_string());

    // Compute hash using DefaultHasher (fast, good enough for cache keys)
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    head_sha.hash(&mut hasher);
    mtime.hash(&mut hasher);

    Ok(format!("{:016x}", hasher.finish()))
}

/// Get git HEAD commit SHA
fn get_git_head(repo_path: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(anyhow!("git rev-parse failed"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ============================================================================
// PUB-004: Git Status Parsing
// ============================================================================

/// Get git status for a repo
pub fn get_git_status(repo_path: &Path) -> Result<GitStatus> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(anyhow!("git status failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut status = GitStatus::default();

    for line in stdout.lines() {
        if line.len() < 2 {
            continue;
        }
        let index = line.chars().next().unwrap_or(' ');
        let worktree = line.chars().nth(1).unwrap_or(' ');

        match (index, worktree) {
            ('?', '?') => status.untracked += 1,
            (i, _) if i != ' ' && i != '?' => status.staged += 1,
            (_, w) if w != ' ' && w != '?' => status.modified += 1,
            _ => {}
        }
    }

    status.is_clean = status.total_changes() == 0;
    status.head_sha = get_git_head(repo_path).unwrap_or_default();

    Ok(status)
}

// ============================================================================
// PUB-005: Version Parsing
// ============================================================================

/// Extract version from Cargo.toml
pub fn get_local_version(repo_path: &Path) -> Result<String> {
    let cargo_toml = repo_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml)?;

    for line in content.lines() {
        if line.starts_with("version") {
            if let Some(version) = line.split('"').nth(1) {
                return Ok(version.to_string());
            }
        }
    }

    Err(anyhow!("No version found in Cargo.toml"))
}

/// Compare versions and determine action
pub fn determine_action(
    local: Option<&str>,
    crates_io: Option<&str>,
    git_status: &GitStatus,
) -> PublishAction {
    match (local, crates_io) {
        (None, _) => PublishAction::Error,
        (Some(_), None) => {
            if git_status.is_clean {
                PublishAction::NotPublished
            } else {
                PublishAction::NeedsCommit
            }
        }
        (Some(local), Some(remote)) => {
            if !git_status.is_clean {
                PublishAction::NeedsCommit
            } else if local == remote {
                PublishAction::UpToDate
            } else {
                // Parse and compare versions
                match (
                    semver::Version::parse(local),
                    semver::Version::parse(remote),
                ) {
                    (Ok(l), Ok(r)) if l > r => PublishAction::NeedsPublish,
                    (Ok(l), Ok(r)) if l < r => PublishAction::LocalBehind,
                    _ => PublishAction::UpToDate,
                }
            }
        }
    }
}

// ============================================================================
// PUB-006: Scanner Implementation
// ============================================================================

/// Scan workspace for PAIML crates and return publish status
pub struct PublishStatusScanner {
    /// Workspace root (parent of crate directories)
    workspace_root: PathBuf,
    /// Cache
    cache: PublishStatusCache,
    /// crates.io client (for async fetches)
    #[cfg(feature = "native")]
    crates_io: Option<super::crates_io::CratesIoClient>,
}

impl PublishStatusScanner {
    /// Create scanner for workspace
    #[must_use]
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            cache: PublishStatusCache::load(),
            #[cfg(feature = "native")]
            crates_io: None,
        }
    }

    /// Initialize crates.io client
    #[cfg(feature = "native")]
    pub fn with_crates_io(mut self) -> Self {
        self.crates_io = Some(super::crates_io::CratesIoClient::new().with_persistent_cache());
        self
    }

    /// Find all PAIML crate directories in workspace
    #[must_use]
    pub fn find_crate_dirs(&self) -> Vec<(String, PathBuf)> {
        PAIML_CRATES
            .iter()
            .filter_map(|name| {
                let path = self.workspace_root.join(name);
                if path.join("Cargo.toml").exists() {
                    Some(((*name).to_string(), path))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check single crate status (with cache)
    #[allow(dead_code)] // Public API for external consumers
    pub fn check_crate(&mut self, name: &str, path: &Path) -> CrateStatus {
        // Compute cache key
        let cache_key = match compute_cache_key(path) {
            Ok(key) => key,
            Err(e) => {
                return CrateStatus {
                    name: name.to_string(),
                    local_version: None,
                    crates_io_version: None,
                    git_status: GitStatus::default(),
                    action: PublishAction::Error,
                    path: path.to_path_buf(),
                    error: Some(e.to_string()),
                };
            }
        };

        // Check cache
        if let Some(entry) = self.cache.get(name, &cache_key) {
            if !entry.is_crates_io_stale() {
                // Cache hit - O(1)
                return entry.status.clone();
            }
        }

        // Cache miss - need to refresh
        self.refresh_crate(name, path, &cache_key)
    }

    /// Refresh crate status (cache miss path)
    fn refresh_crate(&mut self, name: &str, path: &Path, cache_key: &str) -> CrateStatus {
        let local_version = get_local_version(path).ok();
        let git_status = get_git_status(path).unwrap_or_default();

        // crates.io version fetched separately (async)
        let crates_io_version = None; // Will be filled by async scan

        let action = determine_action(
            local_version.as_deref(),
            crates_io_version.as_deref(),
            &git_status,
        );

        let status = CrateStatus {
            name: name.to_string(),
            local_version,
            crates_io_version,
            git_status,
            action,
            path: path.to_path_buf(),
            error: None,
        };

        // Update cache
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.cache.insert(
            name.to_string(),
            CacheEntry {
                cache_key: cache_key.to_string(),
                status: status.clone(),
                crates_io_checked_at: now,
                created_at: now,
            },
        );

        status
    }

    /// Scan all crates and return report
    #[cfg(feature = "native")]
    pub async fn scan(&mut self) -> Result<PublishStatusReport> {
        use std::time::Instant;

        let start = Instant::now();
        let crate_dirs = self.find_crate_dirs();
        let mut statuses = Vec::with_capacity(crate_dirs.len());
        let mut cache_hits = 0;

        // First pass: check cache and collect statuses
        for (name, path) in &crate_dirs {
            let cache_key = compute_cache_key(path).unwrap_or_default();

            if let Some(entry) = self.cache.get(name, &cache_key) {
                if !entry.is_crates_io_stale() {
                    cache_hits += 1;
                    statuses.push(entry.status.clone());
                    continue;
                }
            }

            // Need refresh - get local info first
            let mut status = self.refresh_crate(name, path, &cache_key);

            // Fetch crates.io version
            if let Some(ref mut client) = self.crates_io {
                if let Ok(response) = client.get_crate(name).await {
                    status.crates_io_version = Some(response.krate.max_version.clone());
                    status.action = determine_action(
                        status.local_version.as_deref(),
                        status.crates_io_version.as_deref(),
                        &status.git_status,
                    );

                    // Update cache with crates.io version
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    self.cache.insert(
                        name.clone(),
                        CacheEntry {
                            cache_key: cache_key.clone(),
                            status: status.clone(),
                            crates_io_checked_at: now,
                            created_at: now,
                        },
                    );
                }
            }

            statuses.push(status);
        }

        // Save cache
        let _ = self.cache.save();

        let elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(PublishStatusReport::from_statuses(
            statuses, cache_hits, elapsed_ms,
        ))
    }

    /// Synchronous scan (for non-async contexts)
    #[cfg(feature = "native")]
    pub fn scan_sync(&mut self) -> Result<PublishStatusReport> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.scan())
    }
}

// ============================================================================
// PUB-007: Display Formatting
// ============================================================================

/// Format report as text table
#[allow(dead_code)] // Used by examples and re-exported in mod.rs
pub fn format_report_text(report: &PublishStatusReport) -> String {
    use std::fmt::Write;

    let mut out = String::new();

    // Header
    writeln!(
        out,
        "{:<20} {:>10} {:>10} {:>10} {:>12}",
        "Crate", "Local", "crates.io", "Git", "Action"
    )
    .unwrap();
    writeln!(out, "{}", "─".repeat(65)).unwrap();

    // Rows
    for status in &report.crates {
        let local = status.local_version.as_deref().unwrap_or("-");
        let remote = status.crates_io_version.as_deref().unwrap_or("-");
        let git = status.git_status.summary();

        writeln!(
            out,
            "{:<20} {:>10} {:>10} {:>10} {:>2} {:>9}",
            status.name,
            local,
            remote,
            git,
            status.action.symbol(),
            status.action.description()
        )
        .unwrap();
    }

    writeln!(out, "{}", "─".repeat(65)).unwrap();

    // Summary
    writeln!(out).unwrap();
    writeln!(
        out,
        "📊 {} crates: {} publish, {} commit, {} up-to-date",
        report.total, report.needs_publish, report.needs_commit, report.up_to_date
    )
    .unwrap();
    writeln!(
        out,
        "⚡ {}ms (cache: {} hits, {} misses)",
        report.elapsed_ms, report.cache_hits, report.cache_misses
    )
    .unwrap();

    out
}

/// Format report as JSON
pub fn format_report_json(report: &PublishStatusReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PUB-001: PublishAction tests
    // ========================================================================

    #[test]
    fn test_PUB_001_action_symbols() {
        assert_eq!(PublishAction::UpToDate.symbol(), "✓");
        assert_eq!(PublishAction::NeedsCommit.symbol(), "📝");
        assert_eq!(PublishAction::NeedsPublish.symbol(), "📦");
        assert_eq!(PublishAction::LocalBehind.symbol(), "⚠️");
        assert_eq!(PublishAction::NotPublished.symbol(), "🆕");
        assert_eq!(PublishAction::Error.symbol(), "❌");
    }

    #[test]
    fn test_PUB_001_action_descriptions() {
        assert_eq!(PublishAction::UpToDate.description(), "up to date");
        assert_eq!(PublishAction::NeedsPublish.description(), "PUBLISH");
    }

    // ========================================================================
    // PUB-002: GitStatus tests
    // ========================================================================

    #[test]
    fn test_PUB_002_git_status_clean() {
        let status = GitStatus {
            modified: 0,
            untracked: 0,
            staged: 0,
            head_sha: "abc123".to_string(),
            is_clean: true,
        };
        assert_eq!(status.total_changes(), 0);
        assert_eq!(status.summary(), "clean");
    }

    #[test]
    fn test_PUB_002_git_status_dirty() {
        let status = GitStatus {
            modified: 3,
            untracked: 2,
            staged: 1,
            head_sha: "abc123".to_string(),
            is_clean: false,
        };
        assert_eq!(status.total_changes(), 6);
        assert_eq!(status.summary(), "3M 2? 1+");
    }

    #[test]
    fn test_PUB_002_git_status_modified_only() {
        let status = GitStatus {
            modified: 5,
            untracked: 0,
            staged: 0,
            head_sha: "def456".to_string(),
            is_clean: false,
        };
        assert_eq!(status.summary(), "5M");
    }

    // ========================================================================
    // PUB-003: Cache tests
    // ========================================================================

    #[test]
    fn test_PUB_003_cache_entry_stale() {
        let old_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - (20 * 60); // 20 minutes ago

        let entry = CacheEntry {
            cache_key: "test".to_string(),
            status: CrateStatus {
                name: "test".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: old_time,
            created_at: old_time,
        };

        assert!(entry.is_crates_io_stale());
    }

    #[test]
    fn test_PUB_003_cache_entry_fresh() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = CacheEntry {
            cache_key: "test".to_string(),
            status: CrateStatus {
                name: "test".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: now,
            created_at: now,
        };

        assert!(!entry.is_crates_io_stale());
    }

    #[test]
    fn test_PUB_003_cache_hit_miss() {
        let mut cache = PublishStatusCache::default();

        // Miss
        assert!(cache.get("test", "key1").is_none());

        // Insert
        let entry = CacheEntry {
            cache_key: "key1".to_string(),
            status: CrateStatus {
                name: "test".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: None,
                git_status: GitStatus::default(),
                action: PublishAction::NotPublished,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: 0,
            created_at: 0,
        };
        cache.insert("test".to_string(), entry);

        // Hit with same key
        assert!(cache.get("test", "key1").is_some());

        // Miss with different key (invalidation)
        assert!(cache.get("test", "key2").is_none());
    }

    // ========================================================================
    // PUB-004: Action determination tests
    // ========================================================================

    #[test]
    fn test_PUB_004_determine_action_up_to_date() {
        let git = GitStatus {
            is_clean: true,
            ..Default::default()
        };
        let action = determine_action(Some("1.0.0"), Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::UpToDate);
    }

    #[test]
    fn test_PUB_004_determine_action_needs_publish() {
        let git = GitStatus {
            is_clean: true,
            ..Default::default()
        };
        let action = determine_action(Some("1.0.1"), Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::NeedsPublish);
    }

    #[test]
    fn test_PUB_004_determine_action_needs_commit() {
        let git = GitStatus {
            is_clean: false,
            modified: 5,
            ..Default::default()
        };
        let action = determine_action(Some("1.0.1"), Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::NeedsCommit);
    }

    #[test]
    fn test_PUB_004_determine_action_local_behind() {
        let git = GitStatus {
            is_clean: true,
            ..Default::default()
        };
        let action = determine_action(Some("1.0.0"), Some("1.0.1"), &git);
        assert_eq!(action, PublishAction::LocalBehind);
    }

    #[test]
    fn test_PUB_004_determine_action_not_published() {
        let git = GitStatus {
            is_clean: true,
            ..Default::default()
        };
        let action = determine_action(Some("1.0.0"), None, &git);
        assert_eq!(action, PublishAction::NotPublished);
    }

    #[test]
    fn test_PUB_004_determine_action_no_local() {
        let git = GitStatus::default();
        let action = determine_action(None, Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::Error);
    }

    // ========================================================================
    // PUB-005: Report tests
    // ========================================================================

    #[test]
    fn test_PUB_005_report_from_statuses() {
        let statuses = vec![
            CrateStatus {
                name: "a".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            },
            CrateStatus {
                name: "b".to_string(),
                local_version: Some("1.0.1".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::NeedsPublish,
                path: PathBuf::from("."),
                error: None,
            },
            CrateStatus {
                name: "c".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus {
                    modified: 3,
                    is_clean: false,
                    ..Default::default()
                },
                action: PublishAction::NeedsCommit,
                path: PathBuf::from("."),
                error: None,
            },
        ];

        let report = PublishStatusReport::from_statuses(statuses, 2, 50);

        assert_eq!(report.total, 3);
        assert_eq!(report.up_to_date, 1);
        assert_eq!(report.needs_publish, 1);
        assert_eq!(report.needs_commit, 1);
        assert_eq!(report.cache_hits, 2);
        assert_eq!(report.cache_misses, 1);
        assert_eq!(report.elapsed_ms, 50);
    }

    // ========================================================================
    // PUB-006: Formatting tests
    // ========================================================================

    #[test]
    fn test_PUB_006_format_report_text() {
        let statuses = vec![CrateStatus {
            name: "trueno".to_string(),
            local_version: Some("0.8.1".to_string()),
            crates_io_version: Some("0.8.1".to_string()),
            git_status: GitStatus {
                is_clean: true,
                ..Default::default()
            },
            action: PublishAction::UpToDate,
            path: PathBuf::from("."),
            error: None,
        }];

        let report = PublishStatusReport::from_statuses(statuses, 1, 10);
        let text = format_report_text(&report);

        assert!(text.contains("trueno"));
        assert!(text.contains("0.8.1"));
        assert!(text.contains("✓"));
        assert!(text.contains("up to date"));
    }

    #[test]
    fn test_PUB_006_format_report_json() {
        let statuses = vec![CrateStatus {
            name: "test".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus::default(),
            action: PublishAction::UpToDate,
            path: PathBuf::from("."),
            error: None,
        }];

        let report = PublishStatusReport::from_statuses(statuses, 0, 5);
        let json = format_report_json(&report).unwrap();

        assert!(json.contains("\"name\": \"test\""));
        assert!(json.contains("\"total\": 1"));
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// PROPERTY: GitStatus total_changes is sum of components
        #[test]
        fn prop_git_status_total(m in 0usize..100, u in 0usize..100, s in 0usize..100) {
            let status = GitStatus {
                modified: m,
                untracked: u,
                staged: s,
                head_sha: String::new(),
                is_clean: m + u + s == 0,
            };
            prop_assert_eq!(status.total_changes(), m + u + s);
        }

        /// PROPERTY: Clean status has zero changes
        #[test]
        fn prop_clean_status_zero_changes(sha in "[a-f0-9]{7}") {
            let status = GitStatus {
                modified: 0,
                untracked: 0,
                staged: 0,
                head_sha: sha,
                is_clean: true,
            };
            prop_assert_eq!(status.total_changes(), 0);
            prop_assert_eq!(status.summary(), "clean");
        }

        /// PROPERTY: Report counts are consistent
        #[test]
        fn prop_report_counts_consistent(
            up in 0usize..10,
            publish in 0usize..10,
            commit in 0usize..10
        ) {
            let mut statuses = Vec::new();

            for i in 0..up {
                statuses.push(CrateStatus {
                    name: format!("up{}", i),
                    local_version: Some("1.0.0".to_string()),
                    crates_io_version: Some("1.0.0".to_string()),
                    git_status: GitStatus::default(),
                    action: PublishAction::UpToDate,
                    path: PathBuf::from("."),
                    error: None,
                });
            }

            for i in 0..publish {
                statuses.push(CrateStatus {
                    name: format!("pub{}", i),
                    local_version: Some("1.0.1".to_string()),
                    crates_io_version: Some("1.0.0".to_string()),
                    git_status: GitStatus::default(),
                    action: PublishAction::NeedsPublish,
                    path: PathBuf::from("."),
                    error: None,
                });
            }

            for i in 0..commit {
                statuses.push(CrateStatus {
                    name: format!("commit{}", i),
                    local_version: Some("1.0.0".to_string()),
                    crates_io_version: Some("1.0.0".to_string()),
                    git_status: GitStatus { modified: 1, is_clean: false, ..Default::default() },
                    action: PublishAction::NeedsCommit,
                    path: PathBuf::from("."),
                    error: None,
                });
            }

            let report = PublishStatusReport::from_statuses(statuses, 0, 0);

            prop_assert_eq!(report.total, up + publish + commit);
            prop_assert_eq!(report.up_to_date, up);
            prop_assert_eq!(report.needs_publish, publish);
            prop_assert_eq!(report.needs_commit, commit);
        }
    }
}
