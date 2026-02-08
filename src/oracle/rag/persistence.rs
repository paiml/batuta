//! RAG Index Persistence - Section 9.7 of oracle-mode-spec.md
//!
//! Persistent storage for the RAG index at `~/.cache/batuta/rag/`.
//! Uses JSON format with BLAKE3 checksums for integrity validation.
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: Graceful degradation on corruption (rebuild instead of crash)
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
///
/// 1.1.0: Added chunk_contents, stemming, stop words, TF-IDF dense search
pub const INDEX_VERSION: &str = "1.1.0";

/// Cache directory relative to user cache
const CACHE_SUBDIR: &str = "batuta/rag";

/// Manifest filename
const MANIFEST_FILE: &str = "manifest.json";

/// Index filename
const INDEX_FILE: &str = "index.json";

/// Documents filename
const DOCUMENTS_FILE: &str = "documents.json";

/// Fingerprints-only filename (lightweight, for `is_index_current` checks)
const FINGERPRINTS_FILE: &str = "fingerprints.json";

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
    /// Chunk content snippets (first 200 chars) for result display
    #[serde(default)]
    pub chunk_contents: HashMap<String, String>,
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

    /// Save index to disk using two-phase commit
    ///
    /// Writes three files with crash safety:
    /// - **Prepare phase**: Write all `.tmp` files (crash here = old cache intact)
    /// - **Commit phase**: Rename all 3, manifest LAST (crash before manifest
    ///   rename = old manifest still valid or checksum mismatch triggers rebuild)
    ///
    /// Files written:
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

        // Clean up any orphaned .tmp files from a previous crashed save
        self.cleanup_tmp_files();

        // Serialize index and documents
        let index_json = serde_json::to_string_pretty(index)?;
        let docs_json = serde_json::to_string_pretty(docs)?;

        // Serialize fingerprints separately for fast is_index_current checks
        let fingerprints_json = serde_json::to_string_pretty(&docs.fingerprints)?;

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
        let manifest_json = serde_json::to_string_pretty(&manifest)?;

        // Phase 1: Prepare — write all .tmp files (crash here = old cache intact)
        self.prepare_write(INDEX_FILE, index_json.as_bytes())?;
        self.prepare_write(DOCUMENTS_FILE, docs_json.as_bytes())?;
        self.prepare_write(FINGERPRINTS_FILE, fingerprints_json.as_bytes())?;
        self.prepare_write(MANIFEST_FILE, manifest_json.as_bytes())?;

        // Phase 2: Commit — rename all, manifest LAST
        // Crash before manifest rename = old manifest checksums won't match new
        // data files, which triggers graceful rebuild on next load().
        self.commit_rename(INDEX_FILE)?;
        self.commit_rename(DOCUMENTS_FILE)?;
        self.commit_rename(FINGERPRINTS_FILE)?;
        self.commit_rename(MANIFEST_FILE)?;

        Ok(())
    }

    /// Load index from disk
    ///
    /// Returns `None` if no cached index exists or if the cache is corrupted
    /// (IO error, checksum mismatch, invalid JSON). Corruption triggers a
    /// warning to stderr so the caller can rebuild gracefully.
    ///
    /// Returns `Err` only for `VersionMismatch` (incompatible format requires
    /// a code update, not just a re-index).
    pub fn load(
        &self,
    ) -> Result<Option<(PersistedIndex, PersistedDocuments, RagManifest)>, PersistenceError> {
        let manifest_path = self.cache_path.join(MANIFEST_FILE);

        // Check if manifest exists
        if !manifest_path.exists() {
            return Ok(None);
        }

        // Load manifest — graceful on IO/JSON errors
        let manifest_json = match fs::read_to_string(&manifest_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG manifest, will rebuild: {e}");
                return Ok(None);
            }
        };
        let manifest: RagManifest = match serde_json::from_str(&manifest_json) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Warning: corrupt RAG manifest JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        // Validate version (Poka-Yoke) — hard error, needs code update
        self.validate_version(&manifest)?;

        // Load and validate index — graceful on IO/JSON/checksum errors
        let index_json = match fs::read_to_string(self.cache_path.join(INDEX_FILE)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG index file, will rebuild: {e}");
                return Ok(None);
            }
        };
        if let Err(e) = self.validate_checksum(&index_json, manifest.index_checksum, "index.json") {
            eprintln!("Warning: {e}, will rebuild");
            return Ok(None);
        }
        let index: PersistedIndex = match serde_json::from_str(&index_json) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("Warning: corrupt RAG index JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        // Load and validate documents — graceful on IO/JSON/checksum errors
        let docs_json = match fs::read_to_string(self.cache_path.join(DOCUMENTS_FILE)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG documents file, will rebuild: {e}");
                return Ok(None);
            }
        };
        if let Err(e) = self.validate_checksum(&docs_json, manifest.docs_checksum, "documents.json")
        {
            eprintln!("Warning: {e}, will rebuild");
            return Ok(None);
        }
        let docs: PersistedDocuments = match serde_json::from_str(&docs_json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: corrupt RAG documents JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        Ok(Some((index, docs, manifest)))
    }

    /// Load only fingerprints for fast `is_index_current` checks.
    ///
    /// Reads ~KB fingerprints.json instead of ~600MB (index.json + documents.json).
    /// Falls back to full `load()` if fingerprints.json doesn't exist (pre-upgrade cache).
    pub fn load_fingerprints_only(
        &self,
    ) -> Result<Option<HashMap<String, DocumentFingerprint>>, PersistenceError> {
        let fp_path = self.cache_path.join(FINGERPRINTS_FILE);

        if fp_path.exists() {
            let fp_json = match fs::read_to_string(&fp_path) {
                Ok(s) => s,
                Err(_) => return self.load_fingerprints_fallback(),
            };
            match serde_json::from_str(&fp_json) {
                Ok(fps) => return Ok(Some(fps)),
                Err(_) => return self.load_fingerprints_fallback(),
            }
        }

        // Fallback: fingerprints.json doesn't exist (pre-upgrade cache)
        self.load_fingerprints_fallback()
    }

    /// Fallback: extract fingerprints from full documents.json load
    fn load_fingerprints_fallback(
        &self,
    ) -> Result<Option<HashMap<String, DocumentFingerprint>>, PersistenceError> {
        self.load()
            .map(|opt| opt.map(|(_, docs, _)| docs.fingerprints))
    }

    /// Clear cached index
    pub fn clear(&self) -> Result<(), PersistenceError> {
        if self.cache_path.exists() {
            // Remove individual files
            let _ = fs::remove_file(self.cache_path.join(MANIFEST_FILE));
            let _ = fs::remove_file(self.cache_path.join(INDEX_FILE));
            let _ = fs::remove_file(self.cache_path.join(DOCUMENTS_FILE));
            let _ = fs::remove_file(self.cache_path.join(FINGERPRINTS_FILE));

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

    /// Phase 1: Write data to a `.tmp` file (prepare)
    fn prepare_write(&self, filename: &str, data: &[u8]) -> Result<(), io::Error> {
        let tmp_path = self.cache_path.join(format!("{}.tmp", filename));

        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;

        Ok(())
    }

    /// Phase 2: Rename `.tmp` file to final path (commit)
    fn commit_rename(&self, filename: &str) -> Result<(), io::Error> {
        let tmp_path = self.cache_path.join(format!("{}.tmp", filename));
        let final_path = self.cache_path.join(filename);

        fs::rename(&tmp_path, &final_path)?;

        Ok(())
    }

    /// Remove orphaned `.tmp` files from a previous crashed save
    fn cleanup_tmp_files(&self) {
        for filename in &[MANIFEST_FILE, INDEX_FILE, DOCUMENTS_FILE] {
            let tmp_path = self.cache_path.join(format!("{}.tmp", filename));
            let _ = fs::remove_file(tmp_path);
        }
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
        let mut chunk_contents = HashMap::new();
        chunk_contents.insert(
            "doc1#1".to_string(),
            "SIMD GPU tensor operations".to_string(),
        );
        chunk_contents.insert(
            "doc2#1".to_string(),
            "machine learning algorithms".to_string(),
        );

        PersistedDocuments {
            documents: HashMap::new(),
            fingerprints: HashMap::new(),
            total_chunks: 5,
            chunk_contents,
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

    // RAG-PERSIST-001b: chunk_contents roundtrip
    #[test]
    fn test_chunk_contents_roundtrip() {
        let (persistence, _tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        persistence.save(&index, &docs, sources).unwrap();

        let (_, loaded_docs, _) = persistence.load().unwrap().unwrap();

        assert_eq!(loaded_docs.chunk_contents.len(), 2);
        assert_eq!(
            loaded_docs.chunk_contents.get("doc1#1").unwrap(),
            "SIMD GPU tensor operations"
        );
        assert_eq!(
            loaded_docs.chunk_contents.get("doc2#1").unwrap(),
            "machine learning algorithms"
        );
    }

    // RAG-PERSIST-002: Checksum corruption returns Ok(None) for graceful rebuild
    #[test]
    fn test_checksum_corruption_returns_none() {
        let (persistence, tmp) = test_persistence();

        let index = sample_index();
        let docs = sample_docs();
        let sources = sample_sources();

        // Save valid index
        persistence.save(&index, &docs, sources).unwrap();

        // Corrupt the index file
        let index_path = tmp.path().join(INDEX_FILE);
        fs::write(&index_path, r#"{"corrupted": true}"#).unwrap();

        // Load should return None (graceful degradation)
        let result = persistence.load().unwrap();
        assert!(result.is_none());
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
        let modified = manifest_json.replace("\"1.1.0\"", "\"2.0.0\"");
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

    // RAG-PERSIST-005: Invalid manifest JSON returns Ok(None) for graceful rebuild
    #[test]
    fn test_invalid_json_returns_none() {
        let (persistence, tmp) = test_persistence();

        // Create invalid manifest
        fs::create_dir_all(tmp.path()).unwrap();
        fs::write(tmp.path().join(MANIFEST_FILE), "not valid json").unwrap();

        // Load should return None (graceful degradation)
        let result = persistence.load().unwrap();
        assert!(result.is_none());
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
        let modified = manifest_json.replace("\"1.1.0\"", "\"1.99.0\"");
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

    // RAG-PERSIST-011: Two-phase save leaves no .tmp orphans
    #[test]
    fn test_two_phase_save_no_tmp_orphans() {
        let (persistence, tmp) = test_persistence();

        persistence
            .save(&sample_index(), &sample_docs(), sample_sources())
            .unwrap();

        // No .tmp files should remain after successful save
        let entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .ends_with(".tmp")
            })
            .collect();
        assert!(entries.is_empty(), "Found orphaned .tmp files: {entries:?}");
    }

    // RAG-PERSIST-012: Save overwrites previous cache (proves clear() unnecessary)
    #[test]
    fn test_save_overwrites_previous_cache() {
        let (persistence, _tmp) = test_persistence();

        // Save initial data
        let mut index = sample_index();
        index.avg_doc_length = 1.0;
        persistence
            .save(&index, &sample_docs(), sample_sources())
            .unwrap();

        // Save different data (no clear() call)
        let mut index2 = sample_index();
        index2.avg_doc_length = 99.0;
        persistence
            .save(&index2, &sample_docs(), sample_sources())
            .unwrap();

        // Load should return the second save's data
        let (loaded, _, _) = persistence.load().unwrap().unwrap();
        assert!(
            (loaded.avg_doc_length - 99.0).abs() < f64::EPSILON,
            "Expected 99.0, got {}",
            loaded.avg_doc_length
        );
    }

    // RAG-PERSIST-013: Checksum mismatch returns Ok(None) (Jidoka graceful degradation)
    #[test]
    fn test_checksum_mismatch_graceful() {
        let (persistence, tmp) = test_persistence();

        persistence
            .save(&sample_index(), &sample_docs(), sample_sources())
            .unwrap();

        // Corrupt documents file (manifest checksums won't match)
        fs::write(
            tmp.path().join(DOCUMENTS_FILE),
            r#"{"documents":{},"fingerprints":{},"total_chunks":0}"#,
        )
        .unwrap();

        let result = persistence.load().unwrap();
        assert!(result.is_none(), "Expected None on checksum mismatch");
    }

    // RAG-PERSIST-014: Missing data file returns Ok(None)
    #[test]
    fn test_missing_data_file_returns_none() {
        let (persistence, tmp) = test_persistence();

        persistence
            .save(&sample_index(), &sample_docs(), sample_sources())
            .unwrap();

        // Delete the index file (simulates partial crash)
        fs::remove_file(tmp.path().join(INDEX_FILE)).unwrap();

        let result = persistence.load().unwrap();
        assert!(result.is_none(), "Expected None on missing data file");
    }

    // RAG-PERSIST-015: Orphan .tmp files cleaned on save
    #[test]
    fn test_orphan_tmp_cleaned_on_save() {
        let (persistence, tmp) = test_persistence();

        // Simulate crashed save by creating orphaned .tmp files
        fs::create_dir_all(tmp.path()).unwrap();
        fs::write(tmp.path().join("index.json.tmp"), "orphan").unwrap();
        fs::write(tmp.path().join("documents.json.tmp"), "orphan").unwrap();
        fs::write(tmp.path().join("manifest.json.tmp"), "orphan").unwrap();

        // Save should clean up orphans first
        persistence
            .save(&sample_index(), &sample_docs(), sample_sources())
            .unwrap();

        // No .tmp files should remain
        let tmp_files: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .ends_with(".tmp")
            })
            .collect();
        assert!(tmp_files.is_empty(), "Orphan .tmp files not cleaned up");

        // And the save should be valid
        let result = persistence.load().unwrap();
        assert!(result.is_some(), "Save should produce valid cache");
    }

    // RAG-PERSIST-016: Manifest checksums consistent after save
    #[test]
    fn test_manifest_checksums_consistent() {
        let (persistence, tmp) = test_persistence();

        persistence
            .save(&sample_index(), &sample_docs(), sample_sources())
            .unwrap();

        // Read manifest and data files, verify checksums match
        let manifest_json = fs::read_to_string(tmp.path().join(MANIFEST_FILE)).unwrap();
        let manifest: RagManifest = serde_json::from_str(&manifest_json).unwrap();

        let index_json = fs::read_to_string(tmp.path().join(INDEX_FILE)).unwrap();
        let docs_json = fs::read_to_string(tmp.path().join(DOCUMENTS_FILE)).unwrap();

        assert_eq!(
            blake3_hash(index_json.as_bytes()),
            manifest.index_checksum,
            "Index checksum mismatch"
        );
        assert_eq!(
            blake3_hash(docs_json.as_bytes()),
            manifest.docs_checksum,
            "Documents checksum mismatch"
        );
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
