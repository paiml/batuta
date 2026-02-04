//! Type definitions for crates.io API responses and caching.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Cache entry with TTL
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: Instant,
}

impl<T> CacheEntry<T> {
    pub fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Persistent cache entry (stored on disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCacheEntry {
    /// Cached response data
    pub response: CrateResponse,
    /// Expiration timestamp (Unix epoch seconds)
    pub expires_at: u64,
}

impl PersistentCacheEntry {
    pub fn new(response: CrateResponse, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            response,
            expires_at: now + ttl.as_secs(),
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now >= self.expires_at
    }
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
