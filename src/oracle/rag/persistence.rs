//! RAG Index Persistence - Section 9.7 of oracle-mode-spec.md
//!
//! Persistent storage for the RAG index at `~/.cache/batuta/rag/`.
//! Uses JSON format with BLAKE3 checksums for integrity validation.
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: Stop-on-error during load if checksum fails
//! - **Poka-Yoke**: Version compatibility prevents format mismatches
//! - **Heijunka**: Incremental updates via fingerprint-based invalidation
//! - **Muda**: JSON for debugging, future P2 uses bincode

use super::fingerprint::{blake3_hash, DocumentFingerprint};
use super::types::{Bm25Config, RrfConfig};
use super::IndexedDocument;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Index format version (semver major.minor.patch)
pub const INDEX_VERSION: &str = "1.0.0";

/// Cache directory relative to user cache
const CACHE_SUBDIR: &str = "batuta/rag";

/// Manifest filename
const MANIFEST_FILE: &str = "manifest.json";

/// Index filename
const INDEX_FILE: &str = "index.json";

/// Documents filename
const DOCUMENTS_FILE: &str = "documents.json";

/// Persisted RAG index manifest
///
/// Contains metadata and checksums for integrity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagManifest {
    /// Index format version (semver)
    pub version: String,
    /// BLAKE3 checksum of index.json
    pub index_checksum: [u8; 32],
    /// BLAKE3 checksum of documents.json
    pub docs_checksum: [u8; 32],
    /// Indexed corpus sources
    pub sources: Vec<CorpusSource>,
    /// Unix timestamp when indexed (milliseconds)
    pub indexed_at: u64,
    /// Batuta version that created this index
    pub batuta_version: String,
}

/// Source corpus information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusSource {
    /// Corpus identifier (e.g., "trueno", "hf-ground-truth-corpus")
    pub id: String,
    /// Git commit hash at index time (if available)
    pub commit: Option<String>,
    /// Number of documents indexed from this source
    pub doc_count: usize,
    /// Number of chunks indexed from this source
    pub chunk_count: usize,
}

/// Serializable inverted index state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedIndex {
    /// Inverted index: term -> (doc_id -> term_frequency)
    pub inverted_index: HashMap<String, HashMap<String, usize>>,
    /// Document lengths for BM25
    pub doc_lengths: HashMap<String, usize>,
    /// BM25 configuration
    pub bm25_config: Bm25Config,
    /// RRF configuration
    pub rrf_config: RrfConfig,
    /// Average document length
    pub avg_doc_length: f64,
}

/// Serializable document metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedDocuments {
    /// Documents by ID
    pub documents: HashMap<String, IndexedDocument>,
    /// Fingerprints for change detection
    pub fingerprints: HashMap<String, DocumentFingerprint>,
    /// Total chunks indexed
    pub total_chunks: usize,
}

/// Persistence errors
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Checksum mismatch (Jidoka halt)
    #[error("Checksum mismatch for {file}: expected {expected:x?}, got {actual:x?}")]
    ChecksumMismatch {
        file: String,
        expected: [u8; 32],
        actual: [u8; 32],
    },

    /// Version mismatch
    #[error("Version mismatch: index version {index_version}, expected {expected_version}")]
    VersionMismatch {
        index_version: String,
        expected_version: String,
    },

    /// Cache directory not found
    #[error("Cache directory not found")]
    CacheDirNotFound,

    /// Manifest not found (no cached index)
    #[error("No cached index found")]
    NoCachedIndex,
}

/// RAG index persistence manager
///
/// Handles saving and loading the RAG index to/from disk.
#[derive(Debug)]
pub struct RagPersistence {
    /// Cache path
    cache_path: PathBuf,
}

impl RagPersistence {
    /// Create persistence manager with default cache path
    ///
    /// Default path: `~/.cache/batuta/rag/`
    pub fn new() -> Self {
        Self {
            cache_path: Self::default_cache_path(),
        }
    }

    /// Create persistence manager with custom cache path
    pub fn with_path(path: PathBuf) -> Self {
        Self { cache_path: path }
    }

    /// Get default cache path
    ///
    /// Uses `dirs::cache_dir()` for platform-specific cache location.
    fn default_cache_path() -> PathBuf {
        #[cfg(feature = "native")]
        {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join(CACHE_SUBDIR)
        }
        #[cfg(not(feature = "native"))]
        {
            PathBuf::from(".cache").join(CACHE_SUBDIR)
        }
    }

    /// Get the cache path
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Save index to disk
    ///
    /// Writes three files atomically:
    /// - `manifest.json`: Version and checksums
    /// - `index.json`: Inverted index data
    /// - `documents.json`: Document metadata
    pub fn save(
        &self,
        index: &PersistedIndex,
        docs: &PersistedDocuments,
        sources: Vec<CorpusSource>,
    ) -> Result<(), PersistenceError> {
        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_path)?;

        // Serialize index and documents
        let index_json = serde_json::to_string_pretty(index)?;
        let docs_json = serde_json::to_string_pretty(docs)?;

        // Compute checksums
        let index_checksum = blake3_hash(index_json.as_bytes());
        let docs_checksum = blake3_hash(docs_json.as_bytes());

        // Create manifest
        let manifest = RagManifest {
            version: INDEX_VERSION.to_string(),
            index_checksum,
            docs_checksum,
            sources,
            indexed_at: current_timestamp_ms(),
            batuta_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Write files atomically (write to .tmp, then rename)
        self.atomic_write(INDEX_FILE, index_json.as_bytes())?;
        self.atomic_write(DOCUMENTS_FILE, docs_json.as_bytes())?;
        self.atomic_write(
            MANIFEST_FILE,
            serde_json::to_string_pretty(&manifest)?.as_bytes(),
        )?;

        Ok(())
    }

    /// Load index from disk
    ///
    /// Returns `None` if no cached index exists.
    /// Returns error if index is corrupted (Jidoka halt).
    pub fn load(
        &self,
    ) -> Result<Option<(PersistedIndex, PersistedDocuments, RagManifest)>, PersistenceError> {
        let manifest_path = self.cache_path.join(MANIFEST_FILE);

        // Check if manifest exists
        if !manifest_path.exists() {
            return Ok(None);
        }

        // Load manifest
        let manifest_json = fs::read_to_string(&manifest_path)?;
        let manifest: RagManifest = serde_json::from_str(&manifest_json)?;

        // Validate version (Poka-Yoke)
        self.validate_version(&manifest)?;

        // Load and validate index
        let index_json = fs::read_to_string(self.cache_path.join(INDEX_FILE))?;
        self.validate_checksum(&index_json, manifest.index_checksum, "index.json")?;
        let index: PersistedIndex = serde_json::from_str(&index_json)?;

        // Load and validate documents
        let docs_json = fs::read_to_string(self.cache_path.join(DOCUMENTS_FILE))?;
        self.validate_checksum(&docs_json, manifest.docs_checksum, "documents.json")?;
        let docs: PersistedDocuments = serde_json::from_str(&docs_json)?;

        Ok(Some((index, docs, manifest)))
    }

    /// Clear cached index
    pub fn clear(&self) -> Result<(), PersistenceError> {
        if self.cache_path.exists() {
            // Remove individual files
            let _ = fs::remove_file(self.cache_path.join(MANIFEST_FILE));
            let _ = fs::remove_file(self.cache_path.join(INDEX_FILE));
            let _ = fs::remove_file(self.cache_path.join(DOCUMENTS_FILE));

            // Try to remove directory if empty
            let _ = fs::remove_dir(&self.cache_path);
        }
        Ok(())
    }

    /// Get index statistics without full load
    pub fn stats(&self) -> Result<Option<RagManifest>, PersistenceError> {
        let manifest_path = self.cache_path.join(MANIFEST_FILE);

        if !manifest_path.exists() {
            return Ok(None);
        }

        let manifest_json = fs::read_to_string(&manifest_path)?;
        let manifest: RagManifest = serde_json::from_str(&manifest_json)?;

        Ok(Some(manifest))
    }

    /// Write file atomically
    fn atomic_write(&self, filename: &str, data: &[u8]) -> Result<(), io::Error> {
        let final_path = self.cache_path.join(filename);
        let tmp_path = self.cache_path.join(format!("{}.tmp", filename));

        // Write to temp file
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;

        // Rename atomically
        fs::rename(&tmp_path, &final_path)?;

        Ok(())
    }

    /// Validate version compatibility (Poka-Yoke)
    fn validate_version(&self, manifest: &RagManifest) -> Result<(), PersistenceError> {
        // Parse versions
        let index_parts: Vec<&str> = manifest.version.split('.').collect();
        let expected_parts: Vec<&str> = INDEX_VERSION.split('.').collect();

        // Major version must match for compatibility
        if index_parts.first() != expected_parts.first() {
            return Err(PersistenceError::VersionMismatch {
                index_version: manifest.version.clone(),
                expected_version: INDEX_VERSION.to_string(),
            });
        }

        Ok(())
    }

    /// Validate checksum (Jidoka)
    fn validate_checksum(
        &self,
        data: &str,
        expected: [u8; 32],
        filename: &str,
    ) -> Result<(), PersistenceError> {
        let actual = blake3_hash(data.as_bytes());

        if actual != expected {
            return Err(PersistenceError::ChecksumMismatch {
                file: filename.to_string(),
                expected,
                actual,
            });
        }

        Ok(())
    }
}

impl Default for RagPersistence {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_persistence() -> (RagPersistence, TempDir) {
        let tmp = TempDir::new().unwrap();
        let persistence = RagPersistence::with_path(tmp.path().to_path_buf());
        (persistence, tmp)
    }

    fn sample_index() -> PersistedIndex {
        let mut inverted_index = HashMap::new();
        let mut postings = HashMap::new();
        postings.insert("doc1".to_string(), 3);
        postings.insert("doc2".to_string(), 1);
        inverted_index.insert("test".to_string(), postings);

        let mut doc_lengths = HashMap::new();
        doc_lengths.insert("doc1".to_string(), 10);
        doc_lengths.insert("doc2".to_string(), 5);

        PersistedIndex {
            inverted_index,
            doc_lengths,
            bm25_config: Bm25Config::default(),
            rrf_config: RrfConfig::default(),
            avg_doc_length: 7.5,
        }
    }

    fn sample_docs() -> PersistedDocuments {
        PersistedDocuments {
            documents: HashMap::new(),
            fingerprints: HashMap::new(),
            total_chunks: 5,
        }
    }

    fn sample_sources() -> Vec<CorpusSource> {
        vec![CorpusSource {
            id: "test-corpus".to_string(),
            commit: Some("abc123".to_string()),
            doc_count: 2,
            chunk_count: 5,
        }]
    }

    // RAG-PERSIST-001: Save/load roundtrip
    #[test]
    fn test_save_load_roundtrip() {
        let (persistence, _tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save
        persistence.save(&index, &docs, sources.clone()).unwrap();

        // Load
        let result = persistence.load().unwrap();
        assert!(result.is_some());

        let (loaded_index, loaded_docs, manifest) = result.unwrap();

        // Verify index data
        assert_eq!(loaded_index.avg_doc_length, index.avg_doc_length);
        assert_eq!(loaded_index.doc_lengths.len(), index.doc_lengths.len());
        assert_eq!(
            loaded_index.inverted_index.len(),
            index.inverted_index.len()
        );

        // Verify docs data
        assert_eq!(loaded_docs.total_chunks, docs.total_chunks);

        // Verify manifest
        assert_eq!(manifest.version, INDEX_VERSION);
        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.sources[0].id, "test-corpus");
    }

    // RAG-PERSIST-002: Checksum validation detects corruption
    #[test]
    fn test_checksum_detects_corruption() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save valid index
        persistence.save(&index, &docs, sources).unwrap();

        // Corrupt the index file
        let index_path = tmp.path().join(INDEX_FILE);
        fs::write(&index_path, r#"{"corrupted": true}"#).unwrap();

        // Load should fail with checksum mismatch
        let result = persistence.load();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, PersistenceError::ChecksumMismatch { .. }));
    }

    // RAG-PERSIST-003: Version compatibility check
    #[test]
    fn test_version_compatibility() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save valid index
        persistence.save(&index, &docs, sources).unwrap();

        // Modify manifest to have incompatible version
        let manifest_path = tmp.path().join(MANIFEST_FILE);
        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        let modified = manifest_json.replace("\"1.0.0\"", "\"2.0.0\"");
        fs::write(&manifest_path, modified).unwrap();

        // Load should fail with version mismatch
        let result = persistence.load();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, PersistenceError::VersionMismatch { .. }));
    }

    // RAG-PERSIST-004: Empty cache returns None
    #[test]
    fn test_empty_cache_returns_none() {
        let (persistence, _tmp) = test_persistence();

        let result = persistence.load().unwrap();
        assert!(result.is_none());
    }

    // RAG-PERSIST-005: Graceful degradation on invalid JSON
    #[test]
    fn test_invalid_json_error() {
        let (persistence, tmp) = test_persistence();

        // Create invalid manifest
        fs::create_dir_all(tmp.path()).unwrap();
        fs::write(tmp.path().join(MANIFEST_FILE), "not valid json").unwrap();

        let result = persistence.load();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, PersistenceError::Json(_)));
    }

    // RAG-PERSIST-006: Atomic write prevents partial files
    #[test]
    fn test_atomic_write() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save should not leave .tmp files
        persistence.save(&index, &docs, sources).unwrap();

        // Check no .tmp files exist
        let entries: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
        for entry in entries {
            let path = entry.unwrap().path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            assert!(!filename.ends_with(".tmp"), "Found temp file: {}", filename);
        }
    }

    // RAG-PERSIST-007: Clear removes all cache files
    #[test]
    fn test_clear_removes_files() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save index
        persistence.save(&index, &docs, sources).unwrap();

        // Verify files exist
        assert!(tmp.path().join(MANIFEST_FILE).exists());
        assert!(tmp.path().join(INDEX_FILE).exists());
        assert!(tmp.path().join(DOCUMENTS_FILE).exists());

        // Clear
        persistence.clear().unwrap();

        // Verify files removed
        assert!(!tmp.path().join(MANIFEST_FILE).exists());
        assert!(!tmp.path().join(INDEX_FILE).exists());
        assert!(!tmp.path().join(DOCUMENTS_FILE).exists());
    }

    // RAG-PERSIST-008: Stats returns manifest without full load
    #[test]
    fn test_stats_returns_manifest() {
        let (persistence, _tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        persistence.save(&index, &docs, sources).unwrap();

        let stats = persistence.stats().unwrap();
        assert!(stats.is_some());

        let manifest = stats.unwrap();
        assert_eq!(manifest.version, INDEX_VERSION);
        assert_eq!(manifest.sources.len(), 1);
        assert!(manifest.indexed_at > 0);
    }

    // RAG-PERSIST-009: Default path uses dirs crate
    #[test]
    fn test_default_path() {
        let persistence = RagPersistence::new();
        let path = persistence.cache_path();

        // Should end with batuta/rag
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("batuta") && path_str.contains("rag"),
            "Path should contain batuta/rag: {}",
            path_str
        );
    }

    // RAG-PERSIST-010: Minor version differences are compatible
    #[test]
    fn test_minor_version_compatible() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save valid index
        persistence.save(&index, &docs, sources).unwrap();

        // Modify manifest to have different minor version
        let manifest_path = tmp.path().join(MANIFEST_FILE);
        let manifest_json = fs::read_to_string(&manifest_path).unwrap();
        // Change 1.0.0 to 1.99.0 (same major, different minor)
        let modified = manifest_json.replace("\"1.0.0\"", "\"1.99.0\"");
        fs::write(&manifest_path, modified).unwrap();

        // Also need to update checksum in manifest - or we skip version check
        // For this test, we directly modify the validate_version behavior
        // by checking that only major version matters

        // Actually, let's verify the version parsing logic directly
        let manifest = RagManifest {
            version: "1.99.0".to_string(),
            index_checksum: [0; 32],
            docs_checksum: [0; 32],
            sources: vec![],
            indexed_at: 0,
            batuta_version: "test".to_string(),
        };

        // Should not error - minor version difference is OK
        let result = persistence.validate_version(&manifest);
        assert!(result.is_ok());
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(20))]

            // Property: Roundtrip preserves avg_doc_length
            #[test]
            fn prop_roundtrip_preserves_avg_doc_length(avg in 0.0f64..1000.0) {
                let tmp = TempDir::new().unwrap();
                let persistence = RagPersistence::with_path(tmp.path().to_path_buf());

                let index = PersistedIndex {
                    avg_doc_length: avg,
                    ..Default::default()
                };
                let docs = PersistedDocuments::default();

                persistence.save(&index, &docs, vec![]).unwrap();
                let (loaded, _, _) = persistence.load().unwrap().unwrap();

                prop_assert!((loaded.avg_doc_length - avg).abs() < 1e-10);
            }

            // Property: Checksum is deterministic
            #[test]
            fn prop_checksum_deterministic(data in "[a-z]{10,100}") {
                let hash1 = blake3_hash(data.as_bytes());
                let hash2 = blake3_hash(data.as_bytes());
                prop_assert_eq!(hash1, hash2);
            }

            // Property: Different data produces different checksums
            #[test]
            fn prop_different_data_different_checksum(
                data1 in "[a-z]{10,50}",
                data2 in "[A-Z]{10,50}"
            ) {
                // Only check if data is actually different
                if data1 != data2 {
                    let hash1 = blake3_hash(data1.as_bytes());
                    let hash2 = blake3_hash(data2.as_bytes());
                    prop_assert_ne!(hash1, hash2);
                }
            }

            // Property: Total chunks roundtrips correctly
            #[test]
            fn prop_total_chunks_roundtrip(chunks in 0usize..10000) {
                let tmp = TempDir::new().unwrap();
                let persistence = RagPersistence::with_path(tmp.path().to_path_buf());

                let docs = PersistedDocuments {
                    total_chunks: chunks,
                    ..Default::default()
                };

                persistence.save(&PersistedIndex::default(), &docs, vec![]).unwrap();
                let (_, loaded_docs, _) = persistence.load().unwrap().unwrap();

                prop_assert_eq!(loaded_docs.total_chunks, chunks);
            }
        }
    }
}
