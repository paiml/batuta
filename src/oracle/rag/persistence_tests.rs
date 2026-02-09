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
