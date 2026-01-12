//! HuggingFace Hub API Client
//!
//! Implements HF-QUERY-002 (Hub Search) and HF-QUERY-003 (Asset Metadata)
//!
//! Provides live queries to HuggingFace Hub API:
//! - Model search with filters
//! - Dataset search with filters
//! - Space search with filters
//! - Asset metadata retrieval

// Allow dead_code for now - these types are tested and will be used
// once live Hub API integration is implemented (HUB-API milestone)
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// HF-QUERY-002: Hub Asset Types
// ============================================================================

/// Type of Hub asset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HubAssetType {
    Model,
    Dataset,
    Space,
}

impl std::fmt::Display for HubAssetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => write!(f, "model"),
            Self::Dataset => write!(f, "dataset"),
            Self::Space => write!(f, "space"),
        }
    }
}

/// Hub asset metadata (model, dataset, or space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubAsset {
    /// Asset ID (e.g., "meta-llama/Llama-2-7b-hf")
    pub id: String,
    /// Asset type
    pub asset_type: HubAssetType,
    /// Author/organization
    pub author: String,
    /// Downloads count
    pub downloads: u64,
    /// Likes count
    pub likes: u64,
    /// Tags
    pub tags: Vec<String>,
    /// Pipeline tag (task) - for models
    pub pipeline_tag: Option<String>,
    /// Library (transformers, diffusers, etc.) - for models
    pub library: Option<String>,
    /// License
    pub license: Option<String>,
    /// Last modified timestamp
    pub last_modified: String,
    /// Model card/README content (optional, fetched separately)
    pub card_content: Option<String>,
}

impl HubAsset {
    pub fn new(id: impl Into<String>, asset_type: HubAssetType) -> Self {
        let id_str = id.into();
        let author = id_str.split('/').next().unwrap_or("unknown").to_string();
        Self {
            id: id_str,
            asset_type,
            author,
            downloads: 0,
            likes: 0,
            tags: Vec::new(),
            pipeline_tag: None,
            library: None,
            license: None,
            last_modified: String::new(),
            card_content: None,
        }
    }

    pub fn with_downloads(mut self, downloads: u64) -> Self {
        self.downloads = downloads;
        self
    }

    pub fn with_likes(mut self, likes: u64) -> Self {
        self.likes = likes;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_pipeline_tag(mut self, tag: impl Into<String>) -> Self {
        self.pipeline_tag = Some(tag.into());
        self
    }

    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.library = Some(library.into());
        self
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }
}

// ============================================================================
// HF-QUERY-002: Search Filters
// ============================================================================

/// Search filters for Hub queries
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Filter by task (pipeline_tag)
    pub task: Option<String>,
    /// Filter by library
    pub library: Option<String>,
    /// Filter by author/organization
    pub author: Option<String>,
    /// Filter by license
    pub license: Option<String>,
    /// Minimum downloads threshold
    pub min_downloads: Option<u64>,
    /// Minimum likes threshold
    pub min_likes: Option<u64>,
    /// Search query text
    pub query: Option<String>,
    /// Maximum results to return
    pub limit: usize,
    /// Sort field
    pub sort: Option<String>,
    /// Sort direction (asc/desc)
    pub sort_direction: Option<String>,
}

impl SearchFilters {
    pub fn new() -> Self {
        Self {
            limit: 20,
            ..Default::default()
        }
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.library = Some(library.into());
        self
    }

    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    pub fn with_min_downloads(mut self, min: u64) -> Self {
        self.min_downloads = Some(min);
        self
    }

    pub fn with_min_likes(mut self, min: u64) -> Self {
        self.min_likes = Some(min);
        self
    }

    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_sort(mut self, field: impl Into<String>, direction: impl Into<String>) -> Self {
        self.sort = Some(field.into());
        self.sort_direction = Some(direction.into());
        self
    }
}

// ============================================================================
// HF-QUERY-002/003: Response Cache
// ============================================================================

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    created: Instant,
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            created: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created.elapsed() > self.ttl
    }
}

/// Response cache for Hub queries
#[derive(Debug, Default)]
pub struct ResponseCache {
    search_cache: HashMap<String, CacheEntry<Vec<HubAsset>>>,
    asset_cache: HashMap<String, CacheEntry<HubAsset>>,
    ttl: Duration,
}

impl ResponseCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            search_cache: HashMap::new(),
            asset_cache: HashMap::new(),
            ttl,
        }
    }

    /// Default cache with 15 minute TTL
    pub fn default_ttl() -> Self {
        Self::new(Duration::from_secs(15 * 60))
    }

    /// Cache a search result
    pub fn cache_search(&mut self, key: &str, results: Vec<HubAsset>) {
        self.search_cache
            .insert(key.to_string(), CacheEntry::new(results, self.ttl));
    }

    /// Get cached search result
    pub fn get_search(&self, key: &str) -> Option<&Vec<HubAsset>> {
        self.search_cache.get(key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(&entry.data)
            }
        })
    }

    /// Cache an asset
    pub fn cache_asset(&mut self, id: &str, asset: HubAsset) {
        self.asset_cache
            .insert(id.to_string(), CacheEntry::new(asset, self.ttl));
    }

    /// Get cached asset
    pub fn get_asset(&self, id: &str) -> Option<&HubAsset> {
        self.asset_cache.get(id).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(&entry.data)
            }
        })
    }

    /// Clear expired entries
    pub fn clear_expired(&mut self) {
        self.search_cache.retain(|_, entry| !entry.is_expired());
        self.asset_cache.retain(|_, entry| !entry.is_expired());
    }

    /// Clear all cache
    pub fn clear(&mut self) {
        self.search_cache.clear();
        self.asset_cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            search_entries: self.search_cache.len(),
            asset_entries: self.asset_cache.len(),
            ttl_secs: self.ttl.as_secs(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub search_entries: usize,
    pub asset_entries: usize,
    pub ttl_secs: u64,
}

// ============================================================================
// HF-QUERY-002/003: Hub Client
// ============================================================================

/// HuggingFace Hub API client
#[derive(Debug)]
pub struct HubClient {
    base_url: String,
    cache: ResponseCache,
    offline_mode: bool,
}

impl HubClient {
    /// Create new client with default settings
    pub fn new() -> Self {
        Self {
            base_url: "https://huggingface.co/api".to_string(),
            cache: ResponseCache::default_ttl(),
            offline_mode: false,
        }
    }

    /// Create client with custom base URL (for testing)
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            cache: ResponseCache::default_ttl(),
            offline_mode: false,
        }
    }

    /// Enable offline mode (only return cached data)
    pub fn offline(mut self) -> Self {
        self.offline_mode = true;
        self
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // ========================================================================
    // HF-QUERY-002: Search Methods
    // ========================================================================

    /// Search models on HuggingFace Hub
    pub fn search_models(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("models:{:?}", filters);

        // Check cache first
        if let Some(cached) = self.cache.get_search(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        // In a real implementation, this would make an HTTP request
        // For now, return mock data for testing
        let results = self.mock_model_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        Ok(results)
    }

    /// Search datasets on HuggingFace Hub
    pub fn search_datasets(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("datasets:{:?}", filters);

        if let Some(cached) = self.cache.get_search(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        let results = self.mock_dataset_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        Ok(results)
    }

    /// Search spaces on HuggingFace Hub
    pub fn search_spaces(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("spaces:{:?}", filters);

        if let Some(cached) = self.cache.get_search(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        let results = self.mock_space_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        Ok(results)
    }

    // ========================================================================
    // HF-QUERY-003: Asset Metadata Methods
    // ========================================================================

    /// Get model metadata
    pub fn get_model(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("model:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_model(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        Ok(asset)
    }

    /// Get dataset metadata
    pub fn get_dataset(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("dataset:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_dataset(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        Ok(asset)
    }

    /// Get space metadata
    pub fn get_space(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("space:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            return Ok(cached.clone());
        }

        if self.offline_mode {
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_space(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        Ok(asset)
    }

    // ========================================================================
    // Mock implementations (replace with real API calls)
    // ========================================================================

    fn mock_model_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("meta-llama/Llama-2-7b-hf", HubAssetType::Model)
                .with_downloads(5_000_000)
                .with_likes(10_000)
                .with_pipeline_tag("text-generation")
                .with_library("transformers")
                .with_license("llama2"),
            HubAsset::new("openai/whisper-large-v3", HubAssetType::Model)
                .with_downloads(2_000_000)
                .with_likes(5_000)
                .with_pipeline_tag("automatic-speech-recognition")
                .with_library("transformers")
                .with_license("apache-2.0"),
            HubAsset::new(
                "stabilityai/stable-diffusion-xl-base-1.0",
                HubAssetType::Model,
            )
            .with_downloads(3_000_000)
            .with_likes(8_000)
            .with_pipeline_tag("text-to-image")
            .with_library("diffusers")
            .with_license("openrail++"),
            HubAsset::new(
                "sentence-transformers/all-MiniLM-L6-v2",
                HubAssetType::Model,
            )
            .with_downloads(10_000_000)
            .with_likes(2_000)
            .with_pipeline_tag("sentence-similarity")
            .with_library("sentence-transformers")
            .with_license("apache-2.0"),
            HubAsset::new("bert-base-uncased", HubAssetType::Model)
                .with_downloads(50_000_000)
                .with_likes(15_000)
                .with_pipeline_tag("fill-mask")
                .with_library("transformers")
                .with_license("apache-2.0"),
        ];

        // Apply filters
        if let Some(ref task) = filters.task {
            results.retain(|m| m.pipeline_tag.as_ref().is_some_and(|t| t == task));
        }
        if let Some(ref library) = filters.library {
            results.retain(|m| m.library.as_ref().is_some_and(|l| l == library));
        }
        if let Some(min) = filters.min_downloads {
            results.retain(|m| m.downloads >= min);
        }
        if let Some(min) = filters.min_likes {
            results.retain(|m| m.likes >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_dataset_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("squad", HubAssetType::Dataset)
                .with_downloads(5_000_000)
                .with_likes(1_000)
                .with_tags(vec!["question-answering".into(), "english".into()]),
            HubAsset::new("imdb", HubAssetType::Dataset)
                .with_downloads(3_000_000)
                .with_likes(500)
                .with_tags(vec!["text-classification".into(), "sentiment".into()]),
            HubAsset::new("wikipedia", HubAssetType::Dataset)
                .with_downloads(10_000_000)
                .with_likes(2_000)
                .with_tags(vec!["text".into(), "multilingual".into()]),
        ];

        if let Some(min) = filters.min_downloads {
            results.retain(|d| d.downloads >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_space_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("gradio/chatbot", HubAssetType::Space)
                .with_downloads(100_000)
                .with_likes(500)
                .with_tags(vec!["gradio".into(), "chat".into()]),
            HubAsset::new("stabilityai/stable-diffusion", HubAssetType::Space)
                .with_downloads(500_000)
                .with_likes(2_000)
                .with_tags(vec!["gradio".into(), "image-generation".into()]),
        ];

        if let Some(min) = filters.min_downloads {
            results.retain(|s| s.downloads >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_get_model(&self, id: &str) -> Result<HubAsset, HubError> {
        // Return mock data for known models
        match id {
            "meta-llama/Llama-2-7b-hf" => Ok(HubAsset::new(id, HubAssetType::Model)
                .with_downloads(5_000_000)
                .with_likes(10_000)
                .with_pipeline_tag("text-generation")
                .with_library("transformers")
                .with_license("llama2")),
            "bert-base-uncased" => Ok(HubAsset::new(id, HubAssetType::Model)
                .with_downloads(50_000_000)
                .with_likes(15_000)
                .with_pipeline_tag("fill-mask")
                .with_library("transformers")
                .with_license("apache-2.0")),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }

    fn mock_get_dataset(&self, id: &str) -> Result<HubAsset, HubError> {
        match id {
            "squad" => Ok(HubAsset::new(id, HubAssetType::Dataset)
                .with_downloads(5_000_000)
                .with_likes(1_000)
                .with_tags(vec!["question-answering".into()])),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }

    fn mock_get_space(&self, id: &str) -> Result<HubAsset, HubError> {
        match id {
            "gradio/chatbot" => Ok(HubAsset::new(id, HubAssetType::Space)
                .with_downloads(100_000)
                .with_likes(500)
                .with_tags(vec!["gradio".into(), "chat".into()])),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Hub API error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    /// Asset not found
    NotFound(String),
    /// Rate limited
    RateLimited { retry_after: Option<u64> },
    /// Network error
    NetworkError(String),
    /// Offline mode - no cached data available
    OfflineMode,
    /// Invalid response from API
    InvalidResponse(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Asset not found: {}", id),
            Self::RateLimited { retry_after } => {
                if let Some(secs) = retry_after {
                    write!(f, "Rate limited, retry after {} seconds", secs)
                } else {
                    write!(f, "Rate limited")
                }
            }
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::OfflineMode => write!(f, "Offline mode: no cached data available"),
            Self::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
        }
    }
}

impl std::error::Error for HubError {}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
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
}
