//! RAG indexing commands
//!
//! This module contains the RAG indexing logic, separated from query commands
//! to keep file sizes manageable.

use crate::ansi_colors::Colorize;
use crate::oracle;

use crate::cli::oracle_indexing::{check_dir_for_changes, doc_fingerprint_changed, index_dir_group};

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

/// Print summary stats and save the RAG index to disk.
#[allow(clippy::too_many_arguments)]
fn save_rag_index(
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

/// Index stack documentation for RAG
pub fn cmd_oracle_rag_index(force: bool) -> anyhow::Result<()> {
    use oracle::rag::{
        fingerprint::DocumentFingerprint, persistence::RagPersistence, ChunkerConfig,
        HeijunkaReindexer, HybridRetriever, SemanticChunker,
    };

    println!("{}", "RAG Indexer (Heijunka Mode)".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Initialize persistence
    let persistence = RagPersistence::new();

    // Force rebuild: old cache retained until save overwrites it (crash-safe)
    if force {
        println!(
            "{}",
            "Force rebuild requested (old cache retained until save)...".dimmed()
        );
    }

    // Create chunker configs (needed for fingerprint hashing)
    let rust_chunker_config = ChunkerConfig::new(
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
    );

    let python_chunker_config = ChunkerConfig::new(
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
    );

    // Discover stack repositories (Rust crates)
    let rust_stack_dirs = vec![
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
    ];

    // Rust ground truth corpora (comprehensive MLOps patterns)
    let rust_corpus_dirs = vec![
        "../batuta-ground-truth-mlops-corpus",
        "../apr-model-qa-playbook",
        "../tgi-ground-truth-corpus",
        // Cookbooks and books
        "../batuta-cookbook",
        "../apr-cookbook",
        "../sovereign-ai-book",
        "../sovereign-ai-stack-book",
        "../pmat-book",
    ];

    // Python ground truth corpora (cross-language reference)
    let python_corpus_dirs = vec![
        "../hf-ground-truth-corpus",
        "../jax-ground-truth-corpus",
        "../vllm-ground-truth-corpus",
        "../algorithm-competition-corpus",
        "../databricks-ground-truth-corpus",
    ];

    let model_hash = [0u8; 32]; // BM25 has no model weights

    eprint_phase("Checking fingerprints...");
    if !force
        && is_index_current(
            &persistence,
            &rust_stack_dirs,
            &rust_corpus_dirs,
            &python_corpus_dirs,
            &rust_chunker_config,
            &python_chunker_config,
            model_hash,
        )
    {
        return Ok(());
    }

    let rust_chunker = SemanticChunker::from_config(&rust_chunker_config);
    let python_chunker = SemanticChunker::from_config(&python_chunker_config);

    let mut reindexer = HeijunkaReindexer::new();
    let mut retriever = HybridRetriever::new();

    // Track fingerprints for persistence
    let mut fingerprints: std::collections::HashMap<String, DocumentFingerprint> =
        std::collections::HashMap::new();

    eprint_phase("Indexing Rust stack...");
    println!("{}", "Scanning Rust stack repositories...".dimmed());
    println!();

    let mut indexed_count = 0;
    let mut total_chunks = 0;
    let mut chunk_contents: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Index Rust stack components
    index_dir_group(
        &rust_stack_dirs,
        false,
        &rust_chunker,
        &rust_chunker_config,
        model_hash,
        "rs",
        true,
        true,
        &mut reindexer,
        &mut retriever,
        &mut indexed_count,
        &mut total_chunks,
        &mut fingerprints,
        &mut chunk_contents,
    );

    // Index Python ground truth corpora
    eprint_phase("Indexing Python corpora...");
    println!();
    println!("{}", "Scanning Python ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &python_corpus_dirs,
        true,
        &python_chunker,
        &python_chunker_config,
        model_hash,
        "py",
        false,
        false,
        &mut reindexer,
        &mut retriever,
        &mut indexed_count,
        &mut total_chunks,
        &mut fingerprints,
        &mut chunk_contents,
    );

    // Index Rust ground truth corpora
    eprint_phase("Indexing Rust corpora...");
    println!();
    println!("{}", "Scanning Rust ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &rust_corpus_dirs,
        true,
        &rust_chunker,
        &rust_chunker_config,
        model_hash,
        "rs",
        true,
        true,
        &mut reindexer,
        &mut retriever,
        &mut indexed_count,
        &mut total_chunks,
        &mut fingerprints,
        &mut chunk_contents,
    );

    eprint_phase("Saving index...");
    save_rag_index(
        &persistence,
        retriever,
        reindexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    )
}
