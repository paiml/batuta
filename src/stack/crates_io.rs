#![allow(dead_code)]
//! Crates.io API Client
//!
//! Provides functionality to query crates.io for version information
//! and verify published crate status.
//!
//! Features:
//! - In-memory caching with TTL
//! - Persistent file-based cache for offline mode
//! - Configurable cache TTL

use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    expires_at: Instant,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }

    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Persistent cache entry (stored on disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistentCacheEntry {
    /// Cached response data
    response: CrateResponse,
    /// Expiration timestamp (Unix epoch seconds)
    expires_at: u64,
}

impl PersistentCacheEntry {
    fn new(response: CrateResponse, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            response,
            expires_at: now + ttl.as_secs(),
        }
    }

    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now >= self.expires_at
    }
}

/// Persistent cache stored on disk
#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistentCache {
    entries: HashMap<String, PersistentCacheEntry>,
}

impl PersistentCache {
    /// Get cache file path
    fn cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("batuta")
            .join("crates_io_cache.json")
    }

    /// Load cache from disk
    fn load() -> Self {
        let path = Self::cache_path();
        if path.exists() {
            if let Ok(data) = fs::read_to_string(&path) {
                if let Ok(cache) = serde_json::from_str(&data) {
                    return cache;
                }
            }
        }
        Self::default()
    }

    /// Save cache to disk
    fn save(&self) -> Result<()> {
        let path = Self::cache_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&path, data)?;
        Ok(())
    }

    /// Get cached response
    fn get(&self, name: &str) -> Option<&CrateResponse> {
        self.entries.get(name).and_then(|entry| {
            if !entry.is_expired() {
                Some(&entry.response)
            } else {
                None
            }
        })
    }

    /// Insert response into cache
    fn insert(&mut self, name: String, response: CrateResponse, ttl: Duration) {
        self.entries
            .insert(name, PersistentCacheEntry::new(response, ttl));
    }

    /// Clear expired entries
    fn clear_expired(&mut self) {
        self.entries.retain(|_, entry| !entry.is_expired());
    }
}

/// Client for interacting with crates.io API
#[derive(Debug)]
pub struct CratesIoClient {
    /// HTTP client
    #[cfg(feature = "native")]
    client: reqwest::Client,

    /// In-memory cache for crate info (15 minute TTL)
    cache: HashMap<String, CacheEntry<CrateResponse>>,

    /// Persistent cache for offline mode
    persistent_cache: Option<PersistentCache>,

    /// Cache TTL
    cache_ttl: Duration,

    /// Offline mode - only use cached data
    offline: bool,
}

/// Response from crates.io API for a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateResponse {
    #[serde(rename = "crate")]
    pub krate: CrateData,
    pub versions: Vec<VersionData>,
}

/// Core crate data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateData {
    pub name: String,
    pub max_version: String,
    pub max_stable_version: Option<String>,
    pub description: Option<String>,
    pub downloads: u64,
    pub updated_at: String,
}

impl CrateData {
    /// Create a new CrateData with sensible defaults.
    ///
    /// Sets `max_stable_version` to None, `description` to None,
    /// `downloads` to 0, and `updated_at` to a standard timestamp.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_version: version.into(),
            max_stable_version: None,
            description: None,
            downloads: 0,
            updated_at: "2025-12-05T00:00:00Z".to_string(),
        }
    }
}

/// Version data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionData {
    pub num: String,
    pub yanked: bool,
    pub downloads: u64,
    pub created_at: String,
}

impl VersionData {
    /// Create a new VersionData with sensible defaults (yanked=false, standard timestamp).
    pub fn new(num: impl Into<String>, downloads: u64) -> Self {
        Self {
            num: num.into(),
            yanked: false,
            downloads,
            created_at: "2025-12-05T00:00:00Z".to_string(),
        }
    }
}

/// Response from crates.io dependencies endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResponse {
    pub dependencies: Vec<DependencyData>,
}

/// Individual dependency data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyData {
    pub crate_id: String,
    #[serde(rename = "req")]
    pub version_req: String,
    pub kind: String, // "normal", "dev", "build"
    pub optional: bool,
}

impl CratesIoClient {
    /// Create a new crates.io client
    #[cfg(feature = "native")]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("batuta/0.1 (https://github.com/paiml/batuta)")
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            cache: HashMap::new(),
            persistent_cache: None,
            cache_ttl: Duration::from_secs(15 * 60), // 15 minutes
            offline: false,
        }
    }

    /// Create a client with custom TTL
    #[cfg(feature = "native")]
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Enable persistent file-based cache
    #[cfg(feature = "native")]
    pub fn with_persistent_cache(mut self) -> Self {
        self.persistent_cache = Some(PersistentCache::load());
        self
    }

    /// Set offline mode (only use cached data)
    #[cfg(feature = "native")]
    pub fn set_offline(&mut self, offline: bool) {
        self.offline = offline;
    }

    /// Check if client is in offline mode
    #[cfg(feature = "native")]
    pub fn is_offline(&self) -> bool {
        self.offline
    }

    /// HTTP GET + status check + JSON parse helper.
    ///
    /// Performs a GET request to `url`, checks for 404 / non-success status,
    /// and deserialises the response body into `T`.
    #[cfg(feature = "native")]
    async fn fetch_and_parse<T: DeserializeOwned>(&self, url: &str, context: &str) -> Result<T> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch {}: {}", context, e))?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(anyhow!("'{}' not found on crates.io", context));
        }

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch {}: HTTP {}",
                context,
                response.status()
            ));
        }

        response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse {} response: {}", context, e))
    }

    /// Get crate info from crates.io (cached)
    #[cfg(feature = "native")]
    pub async fn get_crate(&mut self, name: &str) -> Result<CrateResponse> {
        // Check in-memory cache first
        if let Some(entry) = self.cache.get(name) {
            if !entry.is_expired() {
                return Ok(entry.value.clone());
            }
        }

        // Check persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            if let Some(response) = persistent.get(name) {
                // Also add to in-memory cache for faster subsequent access
                self.cache.insert(
                    name.to_string(),
                    CacheEntry::new(response.clone(), self.cache_ttl),
                );
                return Ok(response.clone());
            }
        }

        // In offline mode, return error if not in cache
        if self.offline {
            return Err(anyhow!(
                "Crate '{}' not found in cache (offline mode)",
                name
            ));
        }

        // Fetch from API
        let url = format!("https://crates.io/api/v1/crates/{}", name);
        let crate_response: CrateResponse = self
            .fetch_and_parse(&url, &format!("crate {}", name))
            .await?;

        // Cache the result in memory
        self.cache.insert(
            name.to_string(),
            CacheEntry::new(crate_response.clone(), self.cache_ttl),
        );

        // Also save to persistent cache
        if let Some(ref mut persistent) = self.persistent_cache {
            persistent.insert(name.to_string(), crate_response.clone(), self.cache_ttl);
            let _ = persistent.save(); // Ignore save errors
        }

        Ok(crate_response)
    }

    /// Get the latest version of a crate
    #[cfg(feature = "native")]
    pub async fn get_latest_version(&mut self, name: &str) -> Result<semver::Version> {
        let response = self.get_crate(name).await?;
        response
            .krate
            .max_version
            .parse()
            .map_err(|e| anyhow!("Failed to parse version: {}", e))
    }

    /// Check if a specific version is published
    #[cfg(feature = "native")]
    pub async fn is_version_published(
        &mut self,
        name: &str,
        version: &semver::Version,
    ) -> Result<bool> {
        let response = self.get_crate(name).await?;
        let version_str = version.to_string();

        Ok(response
            .versions
            .iter()
            .any(|v| v.num == version_str && !v.yanked))
    }

    /// Check if a crate exists on crates.io
    #[cfg(feature = "native")]
    pub async fn crate_exists(&mut self, name: &str) -> bool {
        self.get_crate(name).await.is_ok()
    }

    /// Get all published versions of a crate (non-yanked)
    #[cfg(feature = "native")]
    pub async fn get_versions(&mut self, name: &str) -> Result<Vec<semver::Version>> {
        let response = self.get_crate(name).await?;

        let mut versions: Vec<semver::Version> = response
            .versions
            .iter()
            .filter(|v| !v.yanked)
            .filter_map(|v| v.num.parse().ok())
            .collect();

        versions.sort();
        versions.reverse(); // Newest first

        Ok(versions)
    }

    /// Get dependencies for a specific crate version from crates.io
    ///
    /// Fetches from `/api/v1/crates/{name}/{version}/dependencies`
    #[cfg(feature = "native")]
    pub async fn get_dependencies(
        &mut self,
        name: &str,
        version: &str,
    ) -> Result<Vec<DependencyData>> {
        // In offline mode, return error
        if self.offline {
            return Err(anyhow!(
                "Cannot fetch dependencies for {}@{} (offline mode)",
                name,
                version
            ));
        }

        let url = format!(
            "https://crates.io/api/v1/crates/{}/{}/dependencies",
            name, version
        );
        let context = format!("dependencies for {}@{}", name, version);

        let dep_response: DependencyResponse = self.fetch_and_parse(&url, &context).await?;

        Ok(dep_response.dependencies)
    }

    /// Verify a crate is available on crates.io (post-publish check)
    #[cfg(feature = "native")]
    pub async fn verify_available(&mut self, name: &str, version: &semver::Version) -> Result<()> {
        // Clear cache to get fresh data
        self.cache.remove(name);

        let max_attempts = 10;
        let delay = Duration::from_secs(3);

        for attempt in 1..=max_attempts {
            if self.is_version_published(name, version).await? {
                return Ok(());
            }

            if attempt < max_attempts {
                tokio::time::sleep(delay).await;
            }
        }

        Err(anyhow!(
            "Crate {}@{} not available on crates.io after {} attempts",
            name,
            version,
            max_attempts
        ))
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Clear expired cache entries
    pub fn clear_expired(&mut self) {
        self.cache.retain(|_, entry| !entry.is_expired());
    }
}

#[cfg(feature = "native")]
impl Default for CratesIoClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock client for testing without network calls
#[derive(Debug, Default)]
pub struct MockCratesIoClient {
    /// Predefined responses
    responses: HashMap<String, Result<CrateResponse, String>>,
}

impl MockCratesIoClient {
    /// Create a new mock client
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mock response for a crate
    pub fn add_crate(&mut self, name: impl Into<String>, version: impl Into<String>) -> &mut Self {
        let name = name.into();
        let version = version.into();

        let response = CrateResponse {
            krate: CrateData {
                max_stable_version: Some(version.clone()),
                downloads: 1000,
                ..CrateData::new(name.clone(), version.clone())
            },
            versions: vec![VersionData::new(version, 1000)],
        };

        self.responses.insert(name, Ok(response));
        self
    }

    /// Add a "not found" response for a crate
    pub fn add_not_found(&mut self, name: impl Into<String>) -> &mut Self {
        self.responses
            .insert(name.into(), Err("Not found".to_string()));
        self
    }

    /// Get crate (mock implementation)
    pub fn get_crate(&self, name: &str) -> Result<CrateResponse> {
        match self.responses.get(name) {
            Some(Ok(response)) => Ok(response.clone()),
            Some(Err(e)) => Err(anyhow!("{}", e)),
            None => Err(anyhow!("Crate '{}' not found", name)),
        }
    }

    /// Get latest version (mock implementation)
    pub fn get_latest_version(&self, name: &str) -> Result<semver::Version> {
        let response = self.get_crate(name)?;
        response
            .krate
            .max_version
            .parse()
            .map_err(|e| anyhow!("Failed to parse version: {}", e))
    }

    /// Check if version is published (mock implementation)
    pub fn is_version_published(&self, name: &str, version: &semver::Version) -> Result<bool> {
        let response = self.get_crate(name)?;
        let version_str = version.to_string();
        Ok(response
            .versions
            .iter()
            .any(|v| v.num == version_str && !v.yanked))
    }
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Create a simple CrateResponse for testing (no versions, no stable).
    pub fn make_response(name: &str, version: &str) -> CrateResponse {
        CrateResponse {
            krate: CrateData::new(name, version),
            versions: vec![],
        }
    }

    /// Create a full CrateResponse with stable version and version list.
    pub fn make_full_response(
        name: &str,
        version: &str,
        description: Option<&str>,
        downloads: u64,
    ) -> CrateResponse {
        CrateResponse {
            krate: CrateData {
                max_stable_version: Some(version.to_string()),
                description: description.map(|s| s.to_string()),
                downloads,
                ..CrateData::new(name, version)
            },
            versions: vec![VersionData::new(version, downloads)],
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::test_helpers::{make_full_response, make_response};
    use super::*;

    // ============================================================================
    // UNIT TESTS - Fast, focused, deterministic
    // Following bashrs style: ARRANGE/ACT/ASSERT with task IDs
    // ============================================================================

    // ============================================================================
    // CRATES-001: CacheEntry behavior
    // ============================================================================

    /// RED PHASE: Test cache entry creation
    #[test]
    fn test_crates_001_cache_entry_creation() {
        // ARRANGE & ACT
        let entry = CacheEntry::new("test_value", Duration::from_secs(60));

        // ASSERT
        assert_eq!(entry.value, "test_value");
        assert!(!entry.is_expired());
    }

    /// RED PHASE: Test cache entry expiration
    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("value", Duration::from_millis(1));
        assert!(!entry.is_expired());

        std::thread::sleep(Duration::from_millis(10));
        assert!(entry.is_expired());
    }

    /// RED PHASE: Test cache entry with zero TTL
    #[test]
    fn test_crates_001_cache_entry_zero_ttl() {
        // ARRANGE
        let entry = CacheEntry::new(42, Duration::from_secs(0));

        // ASSERT - immediately expired
        assert!(entry.is_expired());
    }

    /// RED PHASE: Test cache entry clone (via CrateResponse)
    #[test]
    fn test_crates_001_cache_entry_with_clone() {
        let response = make_response("test", "1.0.0");

        let entry = CacheEntry::new(response.clone(), Duration::from_secs(60));
        assert_eq!(entry.value.krate.name, "test");
    }

    /// RED PHASE: Test cache entry debug
    #[test]
    fn test_crates_001_cache_entry_debug() {
        let entry = CacheEntry::new("debug_test", Duration::from_secs(60));
        let debug = format!("{:?}", entry);
        assert!(debug.contains("CacheEntry"));
        assert!(debug.contains("debug_test"));
    }

    #[test]
    fn test_mock_client_add_crate() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("trueno", "1.2.0");

        let response = mock.get_crate("trueno").unwrap();
        assert_eq!(response.krate.name, "trueno");
        assert_eq!(response.krate.max_version, "1.2.0");
    }

    #[test]
    fn test_mock_client_not_found() {
        let mock = MockCratesIoClient::new();
        assert!(mock.get_crate("nonexistent").is_err());
    }

    #[test]
    fn test_mock_client_add_not_found() {
        let mut mock = MockCratesIoClient::new();
        mock.add_not_found("broken-crate");

        assert!(mock.get_crate("broken-crate").is_err());
    }

    #[test]
    fn test_mock_client_get_latest_version() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("aprender", "0.8.1");

        let version = mock.get_latest_version("aprender").unwrap();
        assert_eq!(version, semver::Version::new(0, 8, 1));
    }

    #[test]
    fn test_mock_client_version_published() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("trueno", "1.2.0");

        // Published version
        assert!(mock
            .is_version_published("trueno", &semver::Version::new(1, 2, 0))
            .unwrap());

        // Not published version
        assert!(!mock
            .is_version_published("trueno", &semver::Version::new(1, 3, 0))
            .unwrap());
    }

    #[test]
    fn test_mock_client_multiple_crates() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("trueno", "1.2.0")
            .add_crate("aprender", "0.8.1")
            .add_crate("renacer", "0.6.0")
            .add_not_found("nonexistent");

        assert!(mock.get_crate("trueno").is_ok());
        assert!(mock.get_crate("aprender").is_ok());
        assert!(mock.get_crate("renacer").is_ok());
        assert!(mock.get_crate("nonexistent").is_err());
    }

    #[test]
    fn test_crate_response_deserialization() {
        let json = r#"{
            "crate": {
                "name": "trueno",
                "max_version": "1.2.0",
                "max_stable_version": "1.2.0",
                "description": "SIMD tensor library",
                "downloads": 5000,
                "updated_at": "2025-12-01T10:00:00Z"
            },
            "versions": [
                {
                    "num": "1.2.0",
                    "yanked": false,
                    "downloads": 3000,
                    "created_at": "2025-12-01T10:00:00Z"
                },
                {
                    "num": "1.1.0",
                    "yanked": false,
                    "downloads": 2000,
                    "created_at": "2025-11-01T10:00:00Z"
                }
            ]
        }"#;

        let response: CrateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.krate.name, "trueno");
        assert_eq!(response.krate.max_version, "1.2.0");
        assert_eq!(response.versions.len(), 2);
        assert!(!response.versions[0].yanked);
    }

    // ============================================================================
    // CRATES-002: CrateResponse and CrateData tests
    // ============================================================================

    /// RED PHASE: Test CrateResponse clone
    #[test]
    fn test_crates_002_crate_response_clone() {
        let mut response = make_full_response("test", "1.0.0", Some("Test crate"), 1000);
        response.krate.updated_at = "2025-12-05".to_string();
        response.versions[0].created_at = "2025-12-05".to_string();

        let cloned = response.clone();
        assert_eq!(cloned.krate.name, response.krate.name);
        assert_eq!(cloned.versions.len(), response.versions.len());
    }

    /// RED PHASE: Test CrateData debug
    #[test]
    fn test_crates_002_crate_data_debug() {
        let data = CrateData::new("debug-crate", "2.0.0");

        let debug = format!("{:?}", data);
        assert!(debug.contains("CrateData"));
        assert!(debug.contains("debug-crate"));
    }

    /// RED PHASE: Test VersionData with yanked=true
    #[test]
    fn test_crates_002_version_data_yanked() {
        let version = VersionData {
            yanked: true,
            ..VersionData::new("0.1.0", 50)
        };

        assert!(version.yanked);
        assert_eq!(version.num, "0.1.0");
    }

    /// RED PHASE: Test VersionData debug
    #[test]
    fn test_crates_002_version_data_debug() {
        let version = VersionData::new("1.2.3", 999);

        let debug = format!("{:?}", version);
        assert!(debug.contains("VersionData"));
        assert!(debug.contains("1.2.3"));
    }

    // ============================================================================
    // CRATES-003: MockCratesIoClient edge cases
    // ============================================================================

    /// RED PHASE: Test MockCratesIoClient default
    #[test]
    fn test_crates_003_mock_client_default() {
        let mock = MockCratesIoClient::default();
        assert!(mock.get_crate("any").is_err());
    }

    /// RED PHASE: Test MockCratesIoClient debug
    #[test]
    fn test_crates_003_mock_client_debug() {
        let mock = MockCratesIoClient::new();
        let debug = format!("{:?}", mock);
        assert!(debug.contains("MockCratesIoClient"));
    }

    /// RED PHASE: Test MockCratesIoClient chaining
    #[test]
    fn test_crates_003_mock_client_chaining() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("a", "1.0.0")
            .add_crate("b", "2.0.0")
            .add_crate("c", "3.0.0");

        assert_eq!(
            mock.get_latest_version("a").unwrap(),
            semver::Version::new(1, 0, 0)
        );
        assert_eq!(
            mock.get_latest_version("b").unwrap(),
            semver::Version::new(2, 0, 0)
        );
        assert_eq!(
            mock.get_latest_version("c").unwrap(),
            semver::Version::new(3, 0, 0)
        );
    }

    /// RED PHASE: Test version not published
    #[test]
    fn test_crates_003_version_not_published() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("test", "1.0.0");

        // Different version should not be published
        let result = mock.is_version_published("test", &semver::Version::new(2, 0, 0));
        assert!(!result.unwrap());
    }

    /// RED PHASE: Test get_latest_version error
    #[test]
    fn test_crates_003_get_latest_version_error() {
        let mock = MockCratesIoClient::new();
        assert!(mock.get_latest_version("nonexistent").is_err());
    }

    /// RED PHASE: Test is_version_published error
    #[test]
    fn test_crates_003_is_version_published_error() {
        let mock = MockCratesIoClient::new();
        let result = mock.is_version_published("nonexistent", &semver::Version::new(1, 0, 0));
        assert!(result.is_err());
    }

    // ============================================================================
    // CRATES-004: Deserialization edge cases
    // ============================================================================

    /// RED PHASE: Test deserialization with null description
    #[test]
    fn test_crates_004_deserialize_null_description() {
        let json = r#"{
            "crate": {
                "name": "minimal",
                "max_version": "0.1.0",
                "max_stable_version": null,
                "description": null,
                "downloads": 0,
                "updated_at": "2025-01-01T00:00:00Z"
            },
            "versions": []
        }"#;

        let response: CrateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.krate.name, "minimal");
        assert!(response.krate.description.is_none());
        assert!(response.krate.max_stable_version.is_none());
        assert!(response.versions.is_empty());
    }

    /// RED PHASE: Test deserialization with prerelease version
    #[test]
    fn test_crates_004_deserialize_prerelease() {
        let json = r#"{
            "crate": {
                "name": "beta-crate",
                "max_version": "1.0.0-beta.1",
                "max_stable_version": "0.9.0",
                "description": "Beta software",
                "downloads": 100,
                "updated_at": "2025-12-01T00:00:00Z"
            },
            "versions": [
                {"num": "1.0.0-beta.1", "yanked": false, "downloads": 50, "created_at": "2025-12-01T00:00:00Z"},
                {"num": "0.9.0", "yanked": false, "downloads": 50, "created_at": "2025-11-01T00:00:00Z"}
            ]
        }"#;

        let response: CrateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.krate.max_version, "1.0.0-beta.1");
        assert_eq!(response.krate.max_stable_version, Some("0.9.0".to_string()));
    }

    /// RED PHASE: Test deserialization with yanked versions
    #[test]
    fn test_crates_004_deserialize_yanked_versions() {
        let json = r#"{
            "crate": {
                "name": "yanked-crate",
                "max_version": "2.0.0",
                "max_stable_version": "2.0.0",
                "description": null,
                "downloads": 1000,
                "updated_at": "2025-12-01T00:00:00Z"
            },
            "versions": [
                {"num": "2.0.0", "yanked": false, "downloads": 500, "created_at": "2025-12-01T00:00:00Z"},
                {"num": "1.0.0", "yanked": true, "downloads": 500, "created_at": "2025-01-01T00:00:00Z"}
            ]
        }"#;

        let response: CrateResponse = serde_json::from_str(json).unwrap();
        assert!(!response.versions[0].yanked);
        assert!(response.versions[1].yanked);
    }

    // ============================================================================
    // CRATES-005: PersistentCacheEntry tests
    // ============================================================================

    /// RED PHASE: Test persistent cache entry creation
    #[test]
    fn test_crates_005_persistent_cache_entry_creation() {
        let response = make_response("test", "1.0.0");

        let entry = PersistentCacheEntry::new(response, Duration::from_secs(3600));
        assert!(!entry.is_expired());
        assert_eq!(entry.response.krate.name, "test");
    }

    /// RED PHASE: Test persistent cache entry expiration
    #[test]
    fn test_crates_005_persistent_cache_entry_expiration() {
        let response = make_response("expired", "0.1.0");

        // Create entry with zero TTL - should be expired
        let entry = PersistentCacheEntry::new(response, Duration::from_secs(0));
        assert!(entry.is_expired());
    }

    /// RED PHASE: Test persistent cache entry serialization
    #[test]
    fn test_crates_005_persistent_cache_entry_serialization() {
        let mut response = make_full_response("serialize", "2.0.0", Some("A test crate"), 1000);
        response.versions[0].downloads = 500;

        let entry = PersistentCacheEntry::new(response, Duration::from_secs(3600));
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: PersistentCacheEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.response.krate.name, "serialize");
        assert_eq!(deserialized.response.versions.len(), 1);
    }

    // ============================================================================
    // CRATES-006: PersistentCache tests
    // ============================================================================

    /// RED PHASE: Test persistent cache default
    #[test]
    fn test_crates_006_persistent_cache_default() {
        let cache = PersistentCache::default();
        assert!(cache.entries.is_empty());
    }

    /// RED PHASE: Test persistent cache insert and get
    #[test]
    fn test_crates_006_persistent_cache_insert_get() {
        let mut cache = PersistentCache::default();

        let response = make_response("cached", "1.0.0");

        cache.insert("cached".to_string(), response, Duration::from_secs(3600));

        let retrieved = cache.get("cached");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().krate.name, "cached");
    }

    /// RED PHASE: Test persistent cache miss
    #[test]
    fn test_crates_006_persistent_cache_miss() {
        let cache = PersistentCache::default();
        assert!(cache.get("nonexistent").is_none());
    }

    /// RED PHASE: Test persistent cache expired entry
    #[test]
    fn test_crates_006_persistent_cache_expired() {
        let mut cache = PersistentCache::default();

        let response = make_response("expired", "1.0.0");

        // Insert with zero TTL - immediately expired
        cache.insert("expired".to_string(), response, Duration::from_secs(0));

        // Should not be retrievable
        assert!(cache.get("expired").is_none());
    }

    /// RED PHASE: Test persistent cache clear expired
    #[test]
    fn test_crates_006_persistent_cache_clear_expired() {
        let mut cache = PersistentCache::default();

        let valid_response = make_response("valid", "1.0.0");

        let expired_response = make_response("expired", "0.1.0");

        cache.insert(
            "valid".to_string(),
            valid_response,
            Duration::from_secs(3600),
        );
        cache.insert(
            "expired".to_string(),
            expired_response,
            Duration::from_secs(0),
        );

        cache.clear_expired();

        assert_eq!(cache.entries.len(), 1);
        assert!(cache.get("valid").is_some());
    }

    /// RED PHASE: Test persistent cache serialization
    #[test]
    fn test_crates_006_persistent_cache_serialization() {
        let mut cache = PersistentCache::default();

        let response = make_response("serialize", "1.0.0");

        cache.insert("serialize".to_string(), response, Duration::from_secs(3600));

        let json = serde_json::to_string(&cache).unwrap();
        let deserialized: PersistentCache = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.entries.len(), 1);
    }

    // ============================================================================
    // CRATES-007: CratesIoClient tests
    // ============================================================================

    #[test]
    #[cfg(feature = "native")]
    fn test_crates_007_client_default() {
        let client = CratesIoClient::default();
        assert!(client.cache.is_empty());
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_crates_007_client_clear_cache() {
        let mut client = CratesIoClient::new();

        // Insert something into cache
        let response = make_response("test", "1.0.0");
        client.cache.insert(
            "test".to_string(),
            CacheEntry::new(response, Duration::from_secs(3600)),
        );

        assert!(!client.cache.is_empty());
        client.clear_cache();
        assert!(client.cache.is_empty());
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_crates_007_client_clear_expired() {
        let mut client = CratesIoClient::new();

        // Insert valid entry
        let valid_response = make_response("valid", "1.0.0");
        client.cache.insert(
            "valid".to_string(),
            CacheEntry::new(valid_response, Duration::from_secs(3600)),
        );

        // Insert expired entry
        let expired_response = make_response("expired", "0.1.0");
        client.cache.insert(
            "expired".to_string(),
            CacheEntry::new(expired_response, Duration::from_secs(0)),
        );

        assert_eq!(client.cache.len(), 2);
        client.clear_expired();
        assert_eq!(client.cache.len(), 1);
        assert!(client.cache.get("valid").is_some());
    }

    // ============================================================================
    // CRATES-008: MockCratesIoClient tests
    // ============================================================================

    #[test]
    fn test_crates_008_mock_client_default() {
        let mock = MockCratesIoClient::default();
        assert!(mock.responses.is_empty());
    }

    #[test]
    fn test_crates_008_mock_client_new() {
        let mock = MockCratesIoClient::new();
        assert!(mock.responses.is_empty());
    }

    #[test]
    fn test_crates_008_mock_client_add_crate() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("test", "1.0.0");

        assert!(mock.responses.contains_key("test"));
    }

    #[test]
    fn test_crates_008_mock_client_add_not_found() {
        let mut mock = MockCratesIoClient::new();
        mock.add_not_found("broken");

        assert!(mock.responses.contains_key("broken"));
        let response = mock.responses.get("broken").unwrap();
        assert!(response.is_err());
    }

    #[test]
    fn test_crates_008_mock_client_get_crate_success() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("trueno", "0.8.0");

        let result = mock.get_crate("trueno");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().krate.max_version, "0.8.0");
    }

    #[test]
    fn test_crates_008_mock_client_get_crate_error() {
        let mut mock = MockCratesIoClient::new();
        mock.add_not_found("notfound");

        let result = mock.get_crate("notfound");
        assert!(result.is_err());
    }

    #[test]
    fn test_crates_008_mock_client_get_crate_not_configured() {
        let mock = MockCratesIoClient::new();
        let result = mock.get_crate("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_crates_008_mock_client_get_latest_version() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("aprender", "0.17.0");

        let version = mock.get_latest_version("aprender").unwrap();
        assert_eq!(version, semver::Version::new(0, 17, 0));
    }

    #[test]
    fn test_crates_008_mock_client_is_version_published() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("realizar", "0.2.3");

        let published = mock
            .is_version_published("realizar", &semver::Version::new(0, 2, 3))
            .unwrap();
        assert!(published);

        let not_published = mock
            .is_version_published("realizar", &semver::Version::new(0, 3, 0))
            .unwrap();
        assert!(!not_published);
    }

    // ============================================================================
    // CRATES-009: VersionData tests
    // ============================================================================

    #[test]
    fn test_crates_009_version_data_debug() {
        let version = VersionData::new("1.0.0", 1000);
        let debug = format!("{:?}", version);
        assert!(debug.contains("1.0.0"));
    }

    #[test]
    fn test_crates_009_version_data_clone() {
        let version = VersionData {
            yanked: true,
            ..VersionData::new("1.0.0", 500)
        };
        let cloned = version.clone();
        assert_eq!(cloned.num, version.num);
        assert_eq!(cloned.yanked, version.yanked);
    }

    // ============================================================================
    // CRATES-010: CrateData tests
    // ============================================================================

    #[test]
    fn test_crates_010_crate_data_with_all_fields() {
        let data = CrateData {
            max_stable_version: Some("2.0.0".to_string()),
            description: Some("A test crate".to_string()),
            downloads: 10000,
            ..CrateData::new("test", "2.0.0")
        };
        let debug = format!("{:?}", data);
        assert!(debug.contains("test"));
        assert!(debug.contains("2.0.0"));
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::test_helpers::{make_full_response, make_response};
    use super::*;
    use proptest::prelude::*;

    // ============================================================================
    // CRATES-007: CratesIoClient sync method tests
    // ============================================================================

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_clear_cache() {
        let mut client = CratesIoClient::new();

        // Add something to cache manually (via internal testing)
        let response = make_response("test", "1.0.0");

        client.cache.insert(
            "test".to_string(),
            CacheEntry::new(response, Duration::from_secs(3600)),
        );

        assert!(!client.cache.is_empty());
        client.clear_cache();
        assert!(client.cache.is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_clear_expired() {
        let mut client = CratesIoClient::new();

        // Add an expired entry
        let response = make_response("expired", "1.0.0");

        client.cache.insert(
            "expired".to_string(),
            CacheEntry::new(response.clone(), Duration::from_secs(0)),
        );

        // Add a valid entry
        client.cache.insert(
            "valid".to_string(),
            CacheEntry::new(response, Duration::from_secs(3600)),
        );

        assert_eq!(client.cache.len(), 2);
        client.clear_expired();
        assert_eq!(client.cache.len(), 1);
        assert!(client.cache.contains_key("valid"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_with_cache_ttl() {
        let client = CratesIoClient::new().with_cache_ttl(Duration::from_secs(60));
        assert_eq!(client.cache_ttl, Duration::from_secs(60));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_with_persistent_cache() {
        let client = CratesIoClient::new().with_persistent_cache();
        assert!(client.persistent_cache.is_some());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_offline_mode() {
        let mut client = CratesIoClient::new();
        assert!(!client.is_offline());

        client.set_offline(true);
        assert!(client.is_offline());

        client.set_offline(false);
        assert!(!client.is_offline());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_007_client_default() {
        let client = CratesIoClient::default();
        assert!(!client.is_offline());
        assert!(client.cache.is_empty());
    }

    // ============================================================================
    // CRATES-008: PersistentCache file operations
    // ============================================================================

    #[test]
    fn test_crates_008_persistent_cache_path() {
        let path = PersistentCache::cache_path();
        assert!(path.to_string_lossy().contains("batuta"));
        assert!(path.to_string_lossy().contains("crates_io_cache.json"));
    }

    #[test]
    fn test_crates_008_persistent_cache_load_nonexistent() {
        // Loading from nonexistent path should return empty cache
        let cache = PersistentCache::load();
        // Just verify it doesn't panic - cache may or may not have entries
        // depending on previous test runs
        let _ = cache.entries.len();
    }

    #[test]
    fn test_crates_008_persistent_cache_save_load_roundtrip() {
        let temp_dir = std::env::temp_dir().join("batuta_crates_io_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let cache_path = temp_dir.join("test_cache.json");

        let mut cache = PersistentCache::default();
        let mut roundtrip_response = make_full_response("test-crate", "1.0.0", Some("Test"), 100);
        roundtrip_response.krate.updated_at = "2025-01-01".to_string();
        roundtrip_response.versions[0].created_at = "2025-01-01".to_string();
        cache.insert(
            "test-crate".to_string(),
            roundtrip_response,
            Duration::from_secs(3600),
        );

        // Save to custom path
        let data = serde_json::to_string_pretty(&cache).unwrap();
        std::fs::write(&cache_path, data).unwrap();

        // Load and verify
        let loaded_data = std::fs::read_to_string(&cache_path).unwrap();
        let loaded: PersistentCache = serde_json::from_str(&loaded_data).unwrap();
        assert!(loaded.get("test-crate").is_some());
        assert_eq!(loaded.get("test-crate").unwrap().krate.name, "test-crate");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    proptest! {
        /// PROPERTY: CacheEntry with positive TTL starts non-expired
        #[test]
        fn prop_cache_entry_starts_valid(ttl_secs in 1u64..1000) {
            let entry = CacheEntry::new("test", Duration::from_secs(ttl_secs));
            prop_assert!(!entry.is_expired());
        }

        /// PROPERTY: MockCratesIoClient returns consistent versions
        #[test]
        fn prop_mock_client_version_consistency(
            major in 0u64..100,
            minor in 0u64..100,
            patch in 0u64..100
        ) {
            let version_str = format!("{}.{}.{}", major, minor, patch);
            let mut mock = MockCratesIoClient::new();
            mock.add_crate("test", &version_str);

            let latest = mock.get_latest_version("test").unwrap();
            prop_assert_eq!(latest.major, major);
            prop_assert_eq!(latest.minor, minor);
            prop_assert_eq!(latest.patch, patch);
        }

        /// PROPERTY: Added crate is always found
        #[test]
        fn prop_added_crate_always_found(
            name in "[a-z][a-z0-9_-]{0,20}",
            version in "[0-9]+\\.[0-9]+\\.[0-9]+"
        ) {
            let mut mock = MockCratesIoClient::new();
            mock.add_crate(&name, &version);

            let result = mock.get_crate(&name);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap().krate.name, name);
        }

        /// PROPERTY: Not-found crate always errors
        #[test]
        fn prop_not_found_always_errors(
            name in "[a-z][a-z0-9_-]{0,20}"
        ) {
            let mut mock = MockCratesIoClient::new();
            mock.add_not_found(&name);

            let result = mock.get_crate(&name);
            prop_assert!(result.is_err());
        }
    }

    // ============================================================================
    // CRATES-009: Additional PersistentCache edge cases
    // ============================================================================

    #[test]
    fn test_crates_009_persistent_cache_entry_debug() {
        let response = make_response("debug", "1.0.0");
        let entry = PersistentCacheEntry::new(response, Duration::from_secs(60));
        let debug = format!("{:?}", entry);
        assert!(debug.contains("PersistentCacheEntry"));
    }

    #[test]
    fn test_crates_009_persistent_cache_debug() {
        let cache = PersistentCache::default();
        let debug = format!("{:?}", cache);
        assert!(debug.contains("PersistentCache"));
    }

    #[test]
    fn test_crates_009_persistent_cache_clone_entry() {
        let mut response = make_full_response("clone", "1.0.0", Some("test"), 100);
        response.krate.updated_at = "2025-01-01".to_string();
        response.versions[0].downloads = 50;
        response.versions[0].created_at = "2025-01-01".to_string();
        let entry = PersistentCacheEntry::new(response, Duration::from_secs(60));
        let cloned = entry.clone();
        assert_eq!(cloned.response.krate.name, "clone");
        assert_eq!(cloned.expires_at, entry.expires_at);
    }

    #[test]
    fn test_crates_009_persistent_cache_multiple_entries() {
        let mut cache = PersistentCache::default();

        for i in 0..10 {
            let mut response = make_response(&format!("crate-{}", i), &format!("0.{}.0", i));
            response.krate.downloads = i as u64 * 100;
            cache.insert(format!("crate-{}", i), response, Duration::from_secs(3600));
        }

        assert_eq!(cache.entries.len(), 10);
        for i in 0..10 {
            assert!(cache.get(&format!("crate-{}", i)).is_some());
        }
    }

    #[test]
    fn test_crates_009_persistent_cache_overwrite() {
        let mut cache = PersistentCache::default();

        let response1 = make_response("overwrite", "1.0.0");
        cache.insert(
            "overwrite".to_string(),
            response1,
            Duration::from_secs(3600),
        );

        let response2 = make_response("overwrite", "2.0.0");
        cache.insert(
            "overwrite".to_string(),
            response2,
            Duration::from_secs(3600),
        );

        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.get("overwrite").unwrap().krate.max_version, "2.0.0");
    }

    // ============================================================================
    // CRATES-010: CratesIoClient debug and additional tests
    // ============================================================================

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_010_client_debug() {
        let client = CratesIoClient::new();
        let debug = format!("{:?}", client);
        assert!(debug.contains("CratesIoClient"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_010_client_cache_insert_and_clear() {
        let mut client = CratesIoClient::new();

        for i in 0..5 {
            let response = make_response(&format!("test-{}", i), "1.0.0");
            client.cache.insert(
                format!("test-{}", i),
                CacheEntry::new(response, Duration::from_secs(3600)),
            );
        }

        assert_eq!(client.cache.len(), 5);
        client.clear_cache();
        assert!(client.cache.is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_crates_010_client_mixed_ttl() {
        let mut client = CratesIoClient::new();

        // Add entries with different TTLs
        for i in 0..3 {
            let response = make_response(&format!("expired-{}", i), "1.0.0");
            // Expired
            client.cache.insert(
                format!("expired-{}", i),
                CacheEntry::new(response, Duration::from_secs(0)),
            );
        }

        for i in 0..3 {
            let response = make_response(&format!("valid-{}", i), "1.0.0");
            // Valid
            client.cache.insert(
                format!("valid-{}", i),
                CacheEntry::new(response, Duration::from_secs(3600)),
            );
        }

        assert_eq!(client.cache.len(), 6);
        client.clear_expired();
        assert_eq!(client.cache.len(), 3);
    }

    // ============================================================================
    // CRATES-011: Serialization roundtrip tests
    // ============================================================================

    #[test]
    fn test_crates_011_crate_data_serialization() {
        let data = CrateData {
            max_stable_version: Some("1.2.3".to_string()),
            description: Some("A test crate for serialization".to_string()),
            downloads: 12345,
            ..CrateData::new("serialize-test", "1.2.3")
        };

        let json = serde_json::to_string(&data).unwrap();
        let deserialized: CrateData = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, data.name);
        assert_eq!(deserialized.max_version, data.max_version);
        assert_eq!(deserialized.max_stable_version, data.max_stable_version);
        assert_eq!(deserialized.description, data.description);
        assert_eq!(deserialized.downloads, data.downloads);
    }

    #[test]
    fn test_crates_011_version_data_serialization() {
        let version = VersionData {
            yanked: true,
            ..VersionData::new("2.0.0-alpha.1", 5000)
        };

        let json = serde_json::to_string(&version).unwrap();
        let deserialized: VersionData = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.num, version.num);
        assert_eq!(deserialized.yanked, version.yanked);
        assert_eq!(deserialized.downloads, version.downloads);
    }

    #[test]
    fn test_crates_011_full_response_serialization() {
        let mut response = make_full_response("full-test", "3.0.0", Some("Complete test"), 50000);
        response.versions = vec![
            VersionData::new("3.0.0", 30000),
            VersionData::new("2.0.0", 15000),
            VersionData {
                yanked: true,
                ..VersionData::new("1.0.0", 5000)
            },
        ];

        let json = serde_json::to_string_pretty(&response).unwrap();
        let deserialized: CrateResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.krate.name, response.krate.name);
        assert_eq!(deserialized.versions.len(), 3);
        assert!(deserialized.versions[2].yanked);
    }

    // ============================================================================
    // CRATES-012: Mock client edge cases
    // ============================================================================

    #[test]
    fn test_crates_012_mock_invalid_version_parse() {
        let mut mock = MockCratesIoClient::new();
        // Add a crate response directly with an invalid version string
        mock.responses.insert(
            "invalid-version".to_string(),
            Ok(make_response("invalid-version", "not-a-version")),
        );

        let result = mock.get_latest_version("invalid-version");
        assert!(result.is_err());
    }

    #[test]
    fn test_crates_012_mock_is_version_published_with_yanked() {
        let mut mock = MockCratesIoClient::new();
        let mut yanked_response = make_response("yanked-test", "2.0.0");
        yanked_response.versions = vec![
            VersionData::new("2.0.0", 0),
            VersionData {
                yanked: true,
                ..VersionData::new("1.0.0", 0)
            },
        ];
        mock.responses
            .insert("yanked-test".to_string(), Ok(yanked_response));

        // Yanked version should not be considered published
        let yanked = mock
            .is_version_published("yanked-test", &semver::Version::new(1, 0, 0))
            .unwrap();
        assert!(!yanked);

        // Non-yanked version should be published
        let published = mock
            .is_version_published("yanked-test", &semver::Version::new(2, 0, 0))
            .unwrap();
        assert!(published);
    }

    #[test]
    fn test_crates_012_mock_empty_versions() {
        let mut mock = MockCratesIoClient::new();
        mock.responses.insert(
            "no-versions".to_string(),
            Ok(make_response("no-versions", "1.0.0")),
        );

        // No versions means version is not published
        let result = mock
            .is_version_published("no-versions", &semver::Version::new(1, 0, 0))
            .unwrap();
        assert!(!result);
    }
}
