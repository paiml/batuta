//! RAG indexing commands
//!
//! This module contains the RAG indexing logic, separated from query commands
//! to keep file sizes manageable.
//!
//! When the `rag` feature is enabled, indexes into SQLite+FTS5 via
//! `trueno_rag::sqlite::SqliteIndex`. Otherwise falls back to
//! `HybridRetriever` + JSON persistence.

use crate::ansi_colors::Colorize;
use crate::oracle;

use crate::cli::oracle_indexing::{check_dir_for_changes, doc_fingerprint_changed, index_dir_group};
#[cfg(feature = "rag")]
use crate::cli::oracle_indexing::ChunkIndexer;

// ============================================================================
// Helper Functions
// ============================================================================

/// Print a phase progress indicator to stderr.
fn eprint_phase(phase: &str) {
    eprintln!("  {} {}", "[   index]".dimmed(), phase);
}

/// Print a labeled statistic with the label in bright yellow.
fn print_stat(label: &str, value: impl std::fmt::Display) {
    println!("{}: {}", label.bright_yellow(), value);
}

/// Default SQLite database path for the RAG index.
#[cfg(feature = "rag")]
fn sqlite_index_path() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("batuta/rag/index.sqlite")
}

/// Check if the existing RAG index is current (no files changed).
/// Returns true if index is up to date and no rebuild is needed.
fn is_index_current(
    persistence: &oracle::rag::persistence::RagPersistence,
    rust_stack_dirs: &[&str],
    rust_corpus_dirs: &[&str],
    python_corpus_dirs: &[&str],
    rust_config: &oracle::rag::ChunkerConfig,
    python_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
) -> bool {
    let Ok(Some((_, existing_docs, _))) = persistence.load() else {
        return false;
    };
    if existing_docs.fingerprints.is_empty() {
        return false;
    }

    println!(
        "{}",
        "Checking for changes against stored fingerprints...".dimmed()
    );

    let changed = detect_dir_changes(
        rust_stack_dirs,
        rust_corpus_dirs,
        python_corpus_dirs,
        rust_config,
        python_config,
        model_hash,
        &existing_docs.fingerprints,
    );

    if changed == 0 {
        println!(
            "{}",
            "Index is current (no files changed since last index)"
                .bright_green()
                .bold()
        );
        println!();
        return true;
    }

    println!(
        "{} files changed, rebuilding index...",
        changed.to_string().bright_yellow()
    );
    println!();
    false
}

/// Detect changes across all directory groups by checking doc fingerprints and source dirs.
/// Returns the number of changed sources (0 = no changes).
fn detect_dir_changes(
    rust_stack_dirs: &[&str],
    rust_corpus_dirs: &[&str],
    python_corpus_dirs: &[&str],
    rust_config: &oracle::rag::ChunkerConfig,
    python_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> usize {
    use std::path::Path;

    let rust_dirs = rust_stack_dirs
        .iter()
        .chain(rust_corpus_dirs.iter())
        .map(|d| (*d, rust_config, "rs"));
    let python_dirs = python_corpus_dirs.iter().map(|d| (*d, python_config, "py"));

    for (dir, config, ext) in rust_dirs.chain(python_dirs) {
        let path = Path::new(dir);
        let component = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        if path.exists()
            && check_component_changed(path, component, config, model_hash, existing, ext)
        {
            return 1;
        }
    }
    0
}

/// Check if a single file within a component directory has changed fingerprint.
fn check_component_file_changed(
    base_path: &std::path::Path,
    filename: &str,
    component: &str,
    config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> bool {
    let path = base_path.join(filename);
    if path.exists() {
        let doc_id = format!("{}/{}", component, filename);
        return doc_fingerprint_changed(&path, &doc_id, config, model_hash, existing);
    }
    false
}

/// Check if any file in a single component directory has changed.
fn check_component_changed(
    path: &std::path::Path,
    component: &str,
    config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    // Check CLAUDE.md and README.md
    if check_component_file_changed(path, "CLAUDE.md", component, config, model_hash, existing)
        || check_component_file_changed(path, "README.md", component, config, model_hash, existing)
    {
        return true;
    }

    // Check src/ directory
    let src_dir = path.join("src");
    if src_dir.exists() {
        let base = src_dir.parent().unwrap_or(&src_dir);
        if check_dir_for_changes(
            &src_dir, base, component, config, model_hash, existing, extension,
        ) {
            return true;
        }
    }

    false
}

// ============================================================================
// SQLite Indexing Backend (rag feature)
// ============================================================================

/// Chunk indexer that collects chunks per document and flushes them into
/// `SqliteIndex` as batch transactions.
#[cfg(feature = "rag")]
pub(crate) struct SqliteChunkIndexer {
    index: trueno_rag::sqlite::SqliteIndex,
    /// Pending chunks grouped by document: doc_id → Vec<(chunk_id, content)>
    pending: std::collections::HashMap<String, Vec<(String, String)>>,
}

#[cfg(feature = "rag")]
impl SqliteChunkIndexer {
    fn new(index: trueno_rag::sqlite::SqliteIndex) -> Self {
        Self {
            index,
            pending: std::collections::HashMap::new(),
        }
    }

    /// Flush all pending chunks into SQLite.
    fn flush(
        &self,
        fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    ) -> anyhow::Result<()> {
        for (doc_id, chunks) in &self.pending {
            let fp = fingerprints.get(doc_id).map(|fp| {
                (doc_id.as_str(), &fp.content_hash)
            });
            self.index
                .insert_document(doc_id, None, Some(doc_id), "", chunks, fp)
                .map_err(|e| anyhow::anyhow!("SQLite insert failed for {doc_id}: {e}"))?;
        }
        Ok(())
    }

    /// Get reference to underlying SqliteIndex.
    fn index(&self) -> &trueno_rag::sqlite::SqliteIndex {
        &self.index
    }
}

#[cfg(feature = "rag")]
impl ChunkIndexer for SqliteChunkIndexer {
    fn index_chunk(&mut self, chunk_id: &str, content: &str) {
        // chunk_id format: "component/file.rs#line"
        let doc_id = chunk_id.split('#').next().unwrap_or(chunk_id);
        self.pending
            .entry(doc_id.to_string())
            .or_default()
            .push((chunk_id.to_string(), content.to_string()));
    }
}

/// Save the RAG index to SQLite, print stats, and optimize.
#[cfg(feature = "rag")]
fn save_rag_index_sqlite(
    sqlite_indexer: &SqliteChunkIndexer,
    reindexer: &oracle::rag::HeijunkaReindexer,
    indexed_count: usize,
    total_chunks: usize,
    fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "{}: {} documents, {} chunks indexed",
        "Complete".bright_green().bold(),
        indexed_count,
        total_chunks
    );
    println!();

    let reindex_stats = reindexer.stats();
    print_stat(
        "Reindexer",
        format!("{} documents tracked", reindex_stats.tracked_documents),
    );
    println!();

    println!("{}", "Flushing to SQLite...".dimmed());
    {
        let _flush_span = span("sqlite_flush");
        sqlite_indexer.flush(fingerprints)?;
    }

    // Store metadata
    let index = sqlite_indexer.index();
    index
        .set_metadata("batuta_version", env!("CARGO_PKG_VERSION"))
        .map_err(|e| anyhow::anyhow!("Failed to set metadata: {e}"))?;
    index
        .set_metadata(
            "indexed_at",
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis().to_string())
                .unwrap_or_default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to set metadata: {e}"))?;

    // Optimize
    {
        let _opt_span = span("sqlite_optimize");
        index
            .optimize()
            .map_err(|e| anyhow::anyhow!("Optimize failed: {e}"))?;
    }

    let doc_count = index
        .document_count()
        .map_err(|e| anyhow::anyhow!("Count failed: {e}"))?;
    let chunk_count = index
        .chunk_count()
        .map_err(|e| anyhow::anyhow!("Count failed: {e}"))?;

    let db_path = sqlite_index_path();
    let db_size = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "{}: {} documents, {} chunks in SQLite ({:.1} MB)",
        "Saved".bright_green().bold(),
        doc_count,
        chunk_count,
        db_size as f64 / 1_048_576.0,
    );
    println!("  {}: {:?}", "Path".dimmed(), db_path);
    println!();

    Ok(())
}

// ============================================================================
// JSON Persistence Backend (fallback without rag feature)
// ============================================================================

/// Print summary stats and save the RAG index to disk as JSON.
#[allow(clippy::too_many_arguments)]
fn save_rag_index_json(
    persistence: &oracle::rag::persistence::RagPersistence,
    retriever: oracle::rag::HybridRetriever,
    reindexer: oracle::rag::HeijunkaReindexer,
    indexed_count: usize,
    total_chunks: usize,
    fingerprints: std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: std::collections::HashMap<String, String>,
) -> anyhow::Result<()> {
    use oracle::rag::persistence::{CorpusSource, PersistedDocuments};

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "{}: {} documents, {} chunks indexed",
        "Complete".bright_green().bold(),
        indexed_count,
        total_chunks
    );
    println!();

    let stats = retriever.stats();
    print_stat("Vocabulary", format!("{} unique terms", stats.total_terms));
    print_stat(
        "Avg doc length",
        format!("{:.1} tokens", stats.avg_doc_length),
    );
    println!();

    let reindex_stats = reindexer.stats();
    print_stat(
        "Reindexer",
        format!("{} documents tracked", reindex_stats.tracked_documents),
    );
    println!();

    let corpus_sources = vec![CorpusSource {
        id: "sovereign-ai-stack".to_string(),
        commit: None,
        doc_count: indexed_count,
        chunk_count: total_chunks,
    }];

    println!("{}", "Saving index to disk...".dimmed());

    let persisted_index = retriever.to_persisted();
    let persisted_docs = PersistedDocuments {
        documents: std::collections::HashMap::new(),
        fingerprints,
        total_chunks,
        chunk_contents,
    };

    match persistence.save(&persisted_index, &persisted_docs, corpus_sources) {
        Ok(()) => {
            println!(
                "{}: Index saved to {:?}",
                "Saved".bright_green().bold(),
                persistence.cache_path()
            );
        }
        Err(e) => {
            println!("{}: Failed to save index: {}", "Warning".bright_yellow(), e);
        }
    }
    println!();

    Ok(())
}

// ============================================================================
// Public Commands
// ============================================================================

/// Common chunker configs, directory lists, and setup for RAG indexing.
struct IndexConfig {
    rust_chunker_config: oracle::rag::ChunkerConfig,
    python_chunker_config: oracle::rag::ChunkerConfig,
    rust_stack_dirs: Vec<&'static str>,
    rust_corpus_dirs: Vec<&'static str>,
    python_corpus_dirs: Vec<&'static str>,
    model_hash: [u8; 32],
}

impl IndexConfig {
    fn new() -> Self {
        Self {
            rust_chunker_config: oracle::rag::ChunkerConfig::new(
                512,
                64,
                &[
                    "\n## ",
                    "\n### ",
                    "\n#### ",
                    "\nfn ",
                    "\npub fn ",
                    "\nimpl ",
                ],
            ),
            python_chunker_config: oracle::rag::ChunkerConfig::new(
                512,
                64,
                &[
                    "\n## ",
                    "\n### ",
                    "\n#### ",
                    "\ndef ",
                    "\nclass ",
                    "\n    def ",
                    "\nasync def ",
                ],
            ),
            rust_stack_dirs: vec![
                // Core compute
                "../trueno",
                "../trueno-db",
                "../trueno-graph",
                "../trueno-rag",
                "../trueno-viz",
                "../trueno-zram",
                "../trueno-ublk",
                // ML/Training/Inference
                "../aprender",
                "../entrenar",
                "../realizar",
                "../whisper.apr",
                "../alimentar",
                // Distribution/Registry
                "../repartir",
                "../pacha",
                // Simulation/Games/Education
                "../jugar",
                "../simular",
                "../profesor",
                // Transpilers
                "../depyler",
                "../bashrs",
                "../decy",
                // Quality/Tooling/Tracing
                "../apr-qa",
                "../renacer",
                "../paiml-mcp-agent-toolkit",
                "../certeza",
                "../verificar",
                "../probar",
                "../presentar",
                "../cohete",
                "../duende",
                "../pepita",
                "../manzana",
                "../copia",
            ],
            rust_corpus_dirs: vec![
                "../batuta-ground-truth-mlops-corpus",
                "../apr-model-qa-playbook",
                "../tgi-ground-truth-corpus",
                "../batuta-cookbook",
                "../apr-cookbook",
                "../sovereign-ai-book",
                "../sovereign-ai-stack-book",
                "../pmat-book",
            ],
            python_corpus_dirs: vec![
                "../hf-ground-truth-corpus",
                "../jax-ground-truth-corpus",
                "../vllm-ground-truth-corpus",
                "../algorithm-competition-corpus",
                "../databricks-ground-truth-corpus",
            ],
            model_hash: [0u8; 32], // BM25 has no model weights
        }
    }
}

/// Run the indexing pipeline, dispatching to SQLite or JSON based on feature.
fn run_indexing(config: &IndexConfig, force: bool) -> anyhow::Result<()> {
    use oracle::rag::{
        fingerprint::DocumentFingerprint, persistence::RagPersistence, HeijunkaReindexer,
        SemanticChunker,
    };

    let persistence = RagPersistence::new();

    if force {
        println!(
            "{}",
            "Force rebuild requested (old cache retained until save)...".dimmed()
        );
    }

    eprint_phase("Checking fingerprints...");
    if !force
        && is_index_current(
            &persistence,
            &config.rust_stack_dirs,
            &config.rust_corpus_dirs,
            &config.python_corpus_dirs,
            &config.rust_chunker_config,
            &config.python_chunker_config,
            config.model_hash,
        )
    {
        return Ok(());
    }

    let rust_chunker = SemanticChunker::from_config(&config.rust_chunker_config);
    let python_chunker = SemanticChunker::from_config(&config.python_chunker_config);
    let mut reindexer = HeijunkaReindexer::new();
    let mut fingerprints: std::collections::HashMap<String, DocumentFingerprint> =
        std::collections::HashMap::new();
    let mut indexed_count = 0;
    let mut total_chunks = 0;
    let mut chunk_contents: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Dispatch to SQLite or JSON indexer
    #[cfg(feature = "rag")]
    {
        let db_path = sqlite_index_path();
        std::fs::create_dir_all(db_path.parent().unwrap())?;
        if force {
            // Only delete DB on explicit force rebuild
            let _ = std::fs::remove_file(&db_path);
        }
        let sqlite_index = trueno_rag::sqlite::SqliteIndex::open(&db_path)
            .map_err(|e| anyhow::anyhow!("Failed to open SQLite index: {e}"))?;
        let mut sqlite_indexer = SqliteChunkIndexer::new(sqlite_index);

        run_index_phases(
            config,
            &rust_chunker,
            &python_chunker,
            &mut reindexer,
            &mut sqlite_indexer,
            &mut indexed_count,
            &mut total_chunks,
            &mut fingerprints,
            &mut chunk_contents,
        );

        eprint_phase("Saving to SQLite...");
        save_rag_index_sqlite(
            &sqlite_indexer,
            &reindexer,
            indexed_count,
            total_chunks,
            &fingerprints,
        )?;

        // Also save JSON for backwards compatibility during transition
        eprint_phase("Saving JSON fallback...");
        let mut retriever = oracle::rag::HybridRetriever::new();
        for (chunk_id, content) in &chunk_contents {
            retriever.index_document(chunk_id, content);
        }
        let _ = save_rag_index_json(
            &persistence,
            retriever,
            HeijunkaReindexer::new(),
            indexed_count,
            total_chunks,
            fingerprints,
            chunk_contents,
        );
    }

    #[cfg(not(feature = "rag"))]
    {
        let mut retriever = oracle::rag::HybridRetriever::new();

        run_index_phases(
            config,
            &rust_chunker,
            &python_chunker,
            &mut reindexer,
            &mut retriever,
            &mut indexed_count,
            &mut total_chunks,
            &mut fingerprints,
            &mut chunk_contents,
        );

        eprint_phase("Saving index...");
        save_rag_index_json(
            &persistence,
            retriever,
            reindexer,
            indexed_count,
            total_chunks,
            fingerprints,
            chunk_contents,
        )?;
    }

    Ok(())
}

/// Run all three indexing phases (Rust stack, Python corpora, Rust corpora).
#[allow(clippy::too_many_arguments)]
fn run_index_phases(
    config: &IndexConfig,
    rust_chunker: &oracle::rag::SemanticChunker,
    python_chunker: &oracle::rag::SemanticChunker,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn crate::cli::oracle_indexing::ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    // Index Rust stack components
    eprint_phase("Indexing Rust stack...");
    println!("{}", "Scanning Rust stack repositories...".dimmed());
    println!();

    index_dir_group(
        &config.rust_stack_dirs,
        false,
        rust_chunker,
        &config.rust_chunker_config,
        config.model_hash,
        "rs",
        true,
        true,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );

    // Index Python ground truth corpora
    eprint_phase("Indexing Python corpora...");
    println!();
    println!("{}", "Scanning Python ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &config.python_corpus_dirs,
        true,
        python_chunker,
        &config.python_chunker_config,
        config.model_hash,
        "py",
        false,
        false,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );

    // Index Rust ground truth corpora
    eprint_phase("Indexing Rust corpora...");
    println!();
    println!("{}", "Scanning Rust ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &config.rust_corpus_dirs,
        true,
        rust_chunker,
        &config.rust_chunker_config,
        config.model_hash,
        "rs",
        true,
        true,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );
}

/// Index stack documentation for RAG
pub fn cmd_oracle_rag_index(force: bool) -> anyhow::Result<()> {
    println!("{}", "RAG Indexer (Heijunka Mode)".bright_cyan().bold());
    #[cfg(feature = "rag")]
    println!("{}", "(SQLite+FTS5 backend)".dimmed());
    #[cfg(not(feature = "rag"))]
    println!("{}", "(JSON fallback backend)".dimmed());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let config = IndexConfig::new();
    run_indexing(&config, force)
}
