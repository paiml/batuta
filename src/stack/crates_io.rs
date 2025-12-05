//! Crates.io API Client
//!
//! Provides functionality to query crates.io for version information
//! and verify published crate status.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

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

/// Client for interacting with crates.io API
#[derive(Debug)]
pub struct CratesIoClient {
    /// HTTP client
    #[cfg(feature = "native")]
    client: reqwest::Client,

    /// Cache for crate info (15 minute TTL)
    cache: HashMap<String, CacheEntry<CrateResponse>>,

    /// Cache TTL
    cache_ttl: Duration,
}

/// Response from crates.io API for a single crate
#[derive(Debug, Clone, Deserialize)]
pub struct CrateResponse {
    #[serde(rename = "crate")]
    pub krate: CrateData,
    pub versions: Vec<VersionData>,
}

/// Core crate data from crates.io
#[derive(Debug, Clone, Deserialize)]
pub struct CrateData {
    pub name: String,
    pub max_version: String,
    pub max_stable_version: Option<String>,
    pub description: Option<String>,
    pub downloads: u64,
    pub updated_at: String,
}

/// Version data from crates.io
#[derive(Debug, Clone, Deserialize)]
pub struct VersionData {
    pub num: String,
    pub yanked: bool,
    pub downloads: u64,
    pub created_at: String,
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
            cache_ttl: Duration::from_secs(15 * 60), // 15 minutes
        }
    }

    /// Create a client with custom TTL
    #[cfg(feature = "native")]
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Get crate info from crates.io (cached)
    #[cfg(feature = "native")]
    pub async fn get_crate(&mut self, name: &str) -> Result<CrateResponse> {
        // Check cache first
        if let Some(entry) = self.cache.get(name) {
            if !entry.is_expired() {
                return Ok(entry.value.clone());
            }
        }

        // Fetch from API
        let url = format!("https://crates.io/api/v1/crates/{}", name);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch crate {}: {}", name, e))?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(anyhow!("Crate '{}' not found on crates.io", name));
        }

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch crate {}: HTTP {}",
                name,
                response.status()
            ));
        }

        let crate_response: CrateResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse crate response: {}", e))?;

        // Cache the result
        self.cache.insert(
            name.to_string(),
            CacheEntry::new(crate_response.clone(), self.cache_ttl),
        );

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
    pub fn add_crate(
        &mut self,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> &mut Self {
        let name = name.into();
        let version = version.into();

        let response = CrateResponse {
            krate: CrateData {
                name: name.clone(),
                max_version: version.clone(),
                max_stable_version: Some(version.clone()),
                description: None,
                downloads: 1000,
                updated_at: "2025-12-05T00:00:00Z".to_string(),
            },
            versions: vec![VersionData {
                num: version,
                yanked: false,
                downloads: 1000,
                created_at: "2025-12-05T00:00:00Z".to_string(),
            }],
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
#[allow(non_snake_case)]
mod tests {
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
    fn test_CRATES_001_cache_entry_creation() {
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
    fn test_CRATES_001_cache_entry_zero_ttl() {
        // ARRANGE
        let entry = CacheEntry::new(42, Duration::from_secs(0));

        // ASSERT - immediately expired
        assert!(entry.is_expired());
    }

    /// RED PHASE: Test cache entry clone (via CrateResponse)
    #[test]
    fn test_CRATES_001_cache_entry_with_clone() {
        let response = CrateResponse {
            krate: CrateData {
                name: "test".to_string(),
                max_version: "1.0.0".to_string(),
                max_stable_version: None,
                description: None,
                downloads: 0,
                updated_at: "".to_string(),
            },
            versions: vec![],
        };

        let entry = CacheEntry::new(response.clone(), Duration::from_secs(60));
        assert_eq!(entry.value.krate.name, "test");
    }

    /// RED PHASE: Test cache entry debug
    #[test]
    fn test_CRATES_001_cache_entry_debug() {
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
    fn test_CRATES_002_crate_response_clone() {
        let response = CrateResponse {
            krate: CrateData {
                name: "test".to_string(),
                max_version: "1.0.0".to_string(),
                max_stable_version: Some("1.0.0".to_string()),
                description: Some("Test crate".to_string()),
                downloads: 1000,
                updated_at: "2025-12-05".to_string(),
            },
            versions: vec![VersionData {
                num: "1.0.0".to_string(),
                yanked: false,
                downloads: 1000,
                created_at: "2025-12-05".to_string(),
            }],
        };

        let cloned = response.clone();
        assert_eq!(cloned.krate.name, response.krate.name);
        assert_eq!(cloned.versions.len(), response.versions.len());
    }

    /// RED PHASE: Test CrateData debug
    #[test]
    fn test_CRATES_002_crate_data_debug() {
        let data = CrateData {
            name: "debug-crate".to_string(),
            max_version: "2.0.0".to_string(),
            max_stable_version: None,
            description: None,
            downloads: 0,
            updated_at: "".to_string(),
        };

        let debug = format!("{:?}", data);
        assert!(debug.contains("CrateData"));
        assert!(debug.contains("debug-crate"));
    }

    /// RED PHASE: Test VersionData with yanked=true
    #[test]
    fn test_CRATES_002_version_data_yanked() {
        let version = VersionData {
            num: "0.1.0".to_string(),
            yanked: true,
            downloads: 50,
            created_at: "2025-01-01".to_string(),
        };

        assert!(version.yanked);
        assert_eq!(version.num, "0.1.0");
    }

    /// RED PHASE: Test VersionData debug
    #[test]
    fn test_CRATES_002_version_data_debug() {
        let version = VersionData {
            num: "1.2.3".to_string(),
            yanked: false,
            downloads: 999,
            created_at: "2025-12-05".to_string(),
        };

        let debug = format!("{:?}", version);
        assert!(debug.contains("VersionData"));
        assert!(debug.contains("1.2.3"));
    }

    // ============================================================================
    // CRATES-003: MockCratesIoClient edge cases
    // ============================================================================

    /// RED PHASE: Test MockCratesIoClient default
    #[test]
    fn test_CRATES_003_mock_client_default() {
        let mock = MockCratesIoClient::default();
        assert!(mock.get_crate("any").is_err());
    }

    /// RED PHASE: Test MockCratesIoClient debug
    #[test]
    fn test_CRATES_003_mock_client_debug() {
        let mock = MockCratesIoClient::new();
        let debug = format!("{:?}", mock);
        assert!(debug.contains("MockCratesIoClient"));
    }

    /// RED PHASE: Test MockCratesIoClient chaining
    #[test]
    fn test_CRATES_003_mock_client_chaining() {
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
    fn test_CRATES_003_version_not_published() {
        let mut mock = MockCratesIoClient::new();
        mock.add_crate("test", "1.0.0");

        // Different version should not be published
        let result = mock.is_version_published("test", &semver::Version::new(2, 0, 0));
        assert!(!result.unwrap());
    }

    /// RED PHASE: Test get_latest_version error
    #[test]
    fn test_CRATES_003_get_latest_version_error() {
        let mock = MockCratesIoClient::new();
        assert!(mock.get_latest_version("nonexistent").is_err());
    }

    /// RED PHASE: Test is_version_published error
    #[test]
    fn test_CRATES_003_is_version_published_error() {
        let mock = MockCratesIoClient::new();
        let result = mock.is_version_published("nonexistent", &semver::Version::new(1, 0, 0));
        assert!(result.is_err());
    }

    // ============================================================================
    // CRATES-004: Deserialization edge cases
    // ============================================================================

    /// RED PHASE: Test deserialization with null description
    #[test]
    fn test_CRATES_004_deserialize_null_description() {
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
    fn test_CRATES_004_deserialize_prerelease() {
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
    fn test_CRATES_004_deserialize_yanked_versions() {
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
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

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
}
