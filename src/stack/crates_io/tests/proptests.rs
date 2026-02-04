//! Property-based tests for crates_io module.

use super::helpers::{make_full_response, make_response};
use crate::stack::crates_io::cache::PersistentCache;
use crate::stack::crates_io::types::{CacheEntry, PersistentCacheEntry};
use crate::stack::crates_io::MockCratesIoClient;
use proptest::prelude::*;
use std::time::Duration;

// ============================================================================
// CRATES-007: CratesIoClient sync method tests
// ============================================================================

#[cfg(feature = "native")]
#[test]
fn test_crates_007_client_clear_cache() {
    use crate::stack::crates_io::client::CratesIoClient;
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
    use crate::stack::crates_io::client::CratesIoClient;
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
    use crate::stack::crates_io::client::CratesIoClient;
    let client = CratesIoClient::new().with_cache_ttl(Duration::from_secs(60));
    assert_eq!(client.cache_ttl, Duration::from_secs(60));
}

#[cfg(feature = "native")]
#[test]
fn test_crates_007_client_with_persistent_cache() {
    use crate::stack::crates_io::client::CratesIoClient;
    let client = CratesIoClient::new().with_persistent_cache();
    assert!(client.persistent_cache.is_some());
}

#[cfg(feature = "native")]
#[test]
fn test_crates_007_client_offline_mode() {
    use crate::stack::crates_io::client::CratesIoClient;
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
    use crate::stack::crates_io::client::CratesIoClient;
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
    use crate::stack::crates_io::client::CratesIoClient;
    let client = CratesIoClient::new();
    let debug = format!("{:?}", client);
    assert!(debug.contains("CratesIoClient"));
}

#[cfg(feature = "native")]
#[test]
fn test_crates_010_client_cache_insert_and_clear() {
    use crate::stack::crates_io::client::CratesIoClient;
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
    use crate::stack::crates_io::client::CratesIoClient;
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
