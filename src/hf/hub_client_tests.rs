use super::*;

// ========================================================================
// HF-QUERY-002-001: HubAssetType Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_001_asset_type_display() {
    assert_eq!(format!("{}", HubAssetType::Model), "model");
    assert_eq!(format!("{}", HubAssetType::Dataset), "dataset");
    assert_eq!(format!("{}", HubAssetType::Space), "space");
}

#[test]
fn test_HF_QUERY_002_002_asset_type_serialize() {
    let json = serde_json::to_string(&HubAssetType::Model).unwrap();
    assert_eq!(json, "\"model\"");
}

// ========================================================================
// HF-QUERY-002-010: HubAsset Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_010_hub_asset_new() {
    let asset = HubAsset::new("org/model", HubAssetType::Model);
    assert_eq!(asset.id, "org/model");
    assert_eq!(asset.author, "org");
    assert_eq!(asset.asset_type, HubAssetType::Model);
}

#[test]
fn test_HF_QUERY_002_011_hub_asset_with_downloads() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_downloads(1000);
    assert_eq!(asset.downloads, 1000);
}

#[test]
fn test_HF_QUERY_002_012_hub_asset_with_likes() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_likes(100);
    assert_eq!(asset.likes, 100);
}

#[test]
fn test_HF_QUERY_002_013_hub_asset_with_tags() {
    let asset = HubAsset::new("org/model", HubAssetType::Model)
        .with_tags(vec!["nlp".into(), "bert".into()]);
    assert_eq!(asset.tags.len(), 2);
}

#[test]
fn test_HF_QUERY_002_014_hub_asset_with_pipeline_tag() {
    let asset =
        HubAsset::new("org/model", HubAssetType::Model).with_pipeline_tag("text-generation");
    assert_eq!(asset.pipeline_tag, Some("text-generation".to_string()));
}

#[test]
fn test_HF_QUERY_002_015_hub_asset_with_library() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_library("transformers");
    assert_eq!(asset.library, Some("transformers".to_string()));
}

#[test]
fn test_HF_QUERY_002_016_hub_asset_with_license() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_license("apache-2.0");
    assert_eq!(asset.license, Some("apache-2.0".to_string()));
}

#[test]
fn test_HF_QUERY_002_017_hub_asset_serialize() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_downloads(100);
    let json = serde_json::to_string(&asset).unwrap();
    assert!(json.contains("\"id\":\"org/model\""));
    assert!(json.contains("\"downloads\":100"));
}

// ========================================================================
// HF-QUERY-002-020: SearchFilters Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_020_filters_new() {
    let filters = SearchFilters::new();
    assert_eq!(filters.limit, 20);
    assert!(filters.task.is_none());
}

#[test]
fn test_HF_QUERY_002_021_filters_with_task() {
    let filters = SearchFilters::new().with_task("text-generation");
    assert_eq!(filters.task, Some("text-generation".to_string()));
}

#[test]
fn test_HF_QUERY_002_022_filters_with_library() {
    let filters = SearchFilters::new().with_library("transformers");
    assert_eq!(filters.library, Some("transformers".to_string()));
}

#[test]
fn test_HF_QUERY_002_023_filters_with_author() {
    let filters = SearchFilters::new().with_author("meta-llama");
    assert_eq!(filters.author, Some("meta-llama".to_string()));
}

#[test]
fn test_HF_QUERY_002_024_filters_with_min_downloads() {
    let filters = SearchFilters::new().with_min_downloads(10000);
    assert_eq!(filters.min_downloads, Some(10000));
}

#[test]
fn test_HF_QUERY_002_025_filters_with_min_likes() {
    let filters = SearchFilters::new().with_min_likes(100);
    assert_eq!(filters.min_likes, Some(100));
}

#[test]
fn test_HF_QUERY_002_026_filters_with_limit() {
    let filters = SearchFilters::new().with_limit(10);
    assert_eq!(filters.limit, 10);
}

#[test]
fn test_HF_QUERY_002_027_filters_with_sort() {
    let filters = SearchFilters::new().with_sort("downloads", "desc");
    assert_eq!(filters.sort, Some("downloads".to_string()));
    assert_eq!(filters.sort_direction, Some("desc".to_string()));
}

#[test]
fn test_HF_QUERY_002_028_filters_chain() {
    let filters = SearchFilters::new()
        .with_task("text-generation")
        .with_library("transformers")
        .with_min_downloads(1000)
        .with_limit(5);
    assert_eq!(filters.task, Some("text-generation".to_string()));
    assert_eq!(filters.library, Some("transformers".to_string()));
    assert_eq!(filters.min_downloads, Some(1000));
    assert_eq!(filters.limit, 5);
}

// ========================================================================
// HF-QUERY-002-030: ResponseCache Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_030_cache_new() {
    let cache = ResponseCache::new(Duration::from_secs(60));
    let stats = cache.stats();
    assert_eq!(stats.search_entries, 0);
    assert_eq!(stats.asset_entries, 0);
}

#[test]
fn test_HF_QUERY_002_031_cache_default_ttl() {
    let cache = ResponseCache::default_ttl();
    let stats = cache.stats();
    assert_eq!(stats.ttl_secs, 15 * 60);
}

#[test]
fn test_HF_QUERY_002_032_cache_search() {
    let mut cache = ResponseCache::default_ttl();
    let results = vec![HubAsset::new("test", HubAssetType::Model)];
    cache.cache_search("key", results.clone());

    let cached = cache.get_search("key");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().len(), 1);
}

#[test]
fn test_HF_QUERY_002_033_cache_search_miss() {
    let cache = ResponseCache::default_ttl();
    assert!(cache.get_search("nonexistent").is_none());
}

#[test]
fn test_HF_QUERY_002_034_cache_asset() {
    let mut cache = ResponseCache::default_ttl();
    let asset = HubAsset::new("org/model", HubAssetType::Model);
    cache.cache_asset("org/model", asset);

    let cached = cache.get_asset("org/model");
    assert!(cached.is_some());
}

#[test]
fn test_HF_QUERY_002_035_cache_clear() {
    let mut cache = ResponseCache::default_ttl();
    cache.cache_search("key", vec![]);
    cache.cache_asset("id", HubAsset::new("id", HubAssetType::Model));

    cache.clear();
    assert!(cache.get_search("key").is_none());
    assert!(cache.get_asset("id").is_none());
}

#[test]
fn test_HF_QUERY_002_036_cache_expired() {
    let mut cache = ResponseCache::new(Duration::from_millis(1));
    cache.cache_search("key", vec![]);

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(10));

    assert!(cache.get_search("key").is_none());
}

// ========================================================================
// HF-QUERY-002-040: HubClient Search Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_040_client_new() {
    let client = HubClient::new();
    assert!(!client.offline_mode);
}

#[test]
fn test_HF_QUERY_002_041_client_search_models() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_models(&filters).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_HF_QUERY_002_042_client_search_models_with_task_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_task("text-generation");
    let results = client.search_models(&filters).unwrap();
    assert!(results
        .iter()
        .all(|m| m.pipeline_tag == Some("text-generation".to_string())));
}

#[test]
fn test_HF_QUERY_002_043_client_search_models_with_library_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_library("diffusers");
    let results = client.search_models(&filters).unwrap();
    assert!(results
        .iter()
        .all(|m| m.library == Some("diffusers".to_string())));
}

#[test]
fn test_HF_QUERY_002_044_client_search_models_with_min_downloads() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_min_downloads(5_000_000);
    let results = client.search_models(&filters).unwrap();
    assert!(results.iter().all(|m| m.downloads >= 5_000_000));
}

#[test]
fn test_HF_QUERY_002_045_client_search_datasets() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_datasets(&filters).unwrap();
    assert!(!results.is_empty());
    assert!(results
        .iter()
        .all(|d| d.asset_type == HubAssetType::Dataset));
}

#[test]
fn test_HF_QUERY_002_046_client_search_spaces() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_spaces(&filters).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|s| s.asset_type == HubAssetType::Space));
}

#[test]
fn test_HF_QUERY_002_047_client_search_caching() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();

    // First call
    let results1 = client.search_models(&filters).unwrap();
    // Second call should hit cache
    let results2 = client.search_models(&filters).unwrap();

    assert_eq!(results1.len(), results2.len());
}

#[test]
fn test_HF_QUERY_002_048_client_offline_mode() {
    let mut client = HubClient::new().offline();
    let filters = SearchFilters::new();
    let result = client.search_models(&filters);
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

// ========================================================================
// HF-QUERY-003-001: Asset Metadata Tests
// ========================================================================

#[test]
fn test_HF_QUERY_003_001_get_model() {
    let mut client = HubClient::new();
    let result = client.get_model("meta-llama/Llama-2-7b-hf");
    assert!(result.is_ok());
    let model = result.unwrap();
    assert_eq!(model.id, "meta-llama/Llama-2-7b-hf");
    assert_eq!(model.asset_type, HubAssetType::Model);
}

#[test]
fn test_HF_QUERY_003_002_get_model_not_found() {
    let mut client = HubClient::new();
    let result = client.get_model("nonexistent/model");
    assert!(matches!(result, Err(HubError::NotFound(_))));
}

#[test]
fn test_HF_QUERY_003_003_get_dataset() {
    let mut client = HubClient::new();
    let result = client.get_dataset("squad");
    assert!(result.is_ok());
    let dataset = result.unwrap();
    assert_eq!(dataset.id, "squad");
    assert_eq!(dataset.asset_type, HubAssetType::Dataset);
}

#[test]
fn test_HF_QUERY_003_004_get_space() {
    let mut client = HubClient::new();
    let result = client.get_space("gradio/chatbot");
    assert!(result.is_ok());
    let space = result.unwrap();
    assert_eq!(space.id, "gradio/chatbot");
    assert_eq!(space.asset_type, HubAssetType::Space);
}

#[test]
fn test_HF_QUERY_003_005_get_model_caching() {
    let mut client = HubClient::new();

    // First call
    let model1 = client.get_model("bert-base-uncased").unwrap();
    // Second call hits cache
    let model2 = client.get_model("bert-base-uncased").unwrap();

    assert_eq!(model1.id, model2.id);
}

#[test]
fn test_HF_QUERY_003_006_clear_cache() {
    let mut client = HubClient::new();
    client.get_model("bert-base-uncased").unwrap();

    let stats_before = client.cache_stats();
    assert!(stats_before.asset_entries > 0);

    client.clear_cache();
    let stats_after = client.cache_stats();
    assert_eq!(stats_after.asset_entries, 0);
}

// ========================================================================
// HF-QUERY-002/003-050: Error Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_050_error_display_not_found() {
    let err = HubError::NotFound("test".to_string());
    assert_eq!(format!("{}", err), "Asset not found: test");
}

#[test]
fn test_HF_QUERY_002_051_error_display_rate_limited() {
    let err = HubError::RateLimited {
        retry_after: Some(60),
    };
    assert!(format!("{}", err).contains("60 seconds"));
}

#[test]
fn test_HF_QUERY_002_052_error_display_offline() {
    let err = HubError::OfflineMode;
    assert!(format!("{}", err).contains("Offline"));
}

#[test]
fn test_HF_QUERY_002_053_error_display_network() {
    let err = HubError::NetworkError("connection refused".to_string());
    assert!(format!("{}", err).contains("connection refused"));
}
