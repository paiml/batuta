//! RAG Oracle Demo - Retrieval-Augmented Generation for Stack Documentation
//!
//! Demonstrates the APR-Powered RAG Oracle with:
//! - Content-addressable indexing (BLAKE3)
//! - Hybrid retrieval (BM25 + dense)
//! - Heijunka load-leveled reindexing
//! - Jidoka stop-on-error validation
//! - Ground truth corpus integration (Python + Rust)
//!
//! Run with: cargo run --example rag_oracle_demo --features native

use batuta::oracle::rag::{
    ChunkerConfig, DocumentFingerprint, HeijunkaReindexer, HybridRetriever, JidokaIndexValidator,
    RagOracle, SemanticChunker,
};

fn main() {
    println!("🔍 RAG Oracle Demo");
    println!("APR-Powered Retrieval-Augmented Generation for Stack Documentation\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. SEMANTIC CHUNKING (Code-Aware)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create chunker with Rust/Markdown separators
    let chunker_config = ChunkerConfig::new(
        512,
        64,
        &[
            "\n## ",     // Markdown H2
            "\n### ",    // Markdown H3
            "\nfn ",     // Rust function
            "\npub fn ", // Rust public function
            "\nimpl ",   // Rust impl block
        ],
    );
    let chunker = SemanticChunker::from_config(&chunker_config);

    let sample_doc = r#"
# Trueno SIMD Library

## Overview

Trueno provides SIMD-accelerated tensor operations for the Sovereign AI Stack.

## Matrix Operations

### Matrix Multiplication

```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // SIMD-accelerated matrix multiplication
    simd_matmul_kernel(a, b)
}
```

### Element-wise Operations

```rust
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    simd_add_kernel(a, b)
}
```

## GPU Backend

For large matrices, GPU dispatch is automatic when compute > 5× PCIe transfer.
"#;

    let chunks = chunker.split(sample_doc);
    println!(
        "📄 Document chunked into {} semantic chunks:\n",
        chunks.len()
    );

    for (i, chunk) in chunks.iter().enumerate() {
        let preview: String = chunk.content.chars().take(60).collect();
        println!(
            "  Chunk {}: lines {}-{} ({} chars)",
            i + 1,
            chunk.start_line,
            chunk.end_line,
            chunk.content.len()
        );
        println!("    Preview: {}...\n", preview.replace('\n', " "));
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. CONTENT-ADDRESSABLE INDEXING (BLAKE3)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model_hash = [0u8; 32]; // Placeholder embedding model hash
    let fingerprint = DocumentFingerprint::new(sample_doc.as_bytes(), &chunker_config, model_hash);

    println!("📝 Document Fingerprint (Poka-Yoke):");
    println!(
        "  Content hash: {:02x}{:02x}{:02x}{:02x}...",
        fingerprint.content_hash[0],
        fingerprint.content_hash[1],
        fingerprint.content_hash[2],
        fingerprint.content_hash[3]
    );
    println!("  Indexed at: {} ms", fingerprint.indexed_at);
    println!();

    // Simulate content change detection
    let modified_doc = sample_doc.replace("SIMD", "AVX-512");
    let new_fingerprint =
        DocumentFingerprint::new(modified_doc.as_bytes(), &chunker_config, model_hash);

    println!("🔄 Change Detection:");
    println!(
        "  Original hash: {:02x}{:02x}{:02x}{:02x}...",
        fingerprint.content_hash[0],
        fingerprint.content_hash[1],
        fingerprint.content_hash[2],
        fingerprint.content_hash[3]
    );
    println!(
        "  Modified hash: {:02x}{:02x}{:02x}{:02x}...",
        new_fingerprint.content_hash[0],
        new_fingerprint.content_hash[1],
        new_fingerprint.content_hash[2],
        new_fingerprint.content_hash[3]
    );
    println!(
        "  Needs reindex: {}\n",
        fingerprint.needs_reindex(&new_fingerprint)
    );

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. HYBRID RETRIEVAL (BM25 + RRF Fusion)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut retriever = HybridRetriever::new();

    // Index sample documents
    retriever.index_document(
        "trueno/CLAUDE.md",
        "SIMD GPU tensor operations accelerated compute matrix multiplication",
    );
    retriever.index_document(
        "aprender/CLAUDE.md",
        "machine learning algorithms random forest gradient boosting neural networks",
    );
    retriever.index_document(
        "entrenar/CLAUDE.md",
        "training autograd LoRA QLoRA quantization fine-tuning",
    );
    retriever.index_document(
        "realizar/CLAUDE.md",
        "inference GGUF safetensors model serving runtime deployment",
    );

    let stats = retriever.stats();
    println!("📚 Index Statistics:");
    println!("  Documents: {}", stats.total_documents);
    println!("  Unique terms: {}", stats.total_terms);
    println!("  Avg doc length: {:.1} tokens\n", stats.avg_doc_length);

    // Perform retrieval queries
    let queries = [
        "GPU tensor operations",
        "machine learning training",
        "model inference",
    ];

    for query in &queries {
        println!("🔍 Query: \"{}\"", query);
        println!("  (In production, use retriever.retrieve() for RRF-fused results)");
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. HEIJUNKA REINDEXING (Load-Leveled)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut reindexer = HeijunkaReindexer::new();

    // Simulate document staleness with different ages
    reindexer.enqueue("trueno/CLAUDE.md", "trueno/CLAUDE.md".into(), 0); // Fresh
    reindexer.enqueue("aprender/CLAUDE.md", "aprender/CLAUDE.md".into(), 86400); // 1 day old
    reindexer.enqueue("entrenar/CLAUDE.md", "entrenar/CLAUDE.md".into(), 604800); // 1 week old
    reindexer.enqueue("realizar/CLAUDE.md", "realizar/CLAUDE.md".into(), 2592000); // 30 days old

    // Record query patterns (affects priority)
    for _ in 0..5 {
        reindexer.record_query("aprender/CLAUDE.md");
    }
    for _ in 0..3 {
        reindexer.record_query("trueno/CLAUDE.md");
    }

    let reindex_stats = reindexer.stats();
    println!("📊 Reindexer Status (Heijunka):");
    println!("  Queue size: {}", reindex_stats.queue_size);
    println!("  Tracked documents: {}", reindex_stats.tracked_documents);
    println!(
        "  Total queries recorded: {}\n",
        reindex_stats.total_queries
    );

    // Get prioritized batch
    let batch = reindexer.next_batch();
    println!("🔄 Next Reindex Batch (priority-ordered):");
    for task in batch.iter().take(2) {
        println!(
            "  - {} (staleness: {:.2})",
            task.doc_id, task.staleness_score
        );
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. JIDOKA VALIDATION (Stop-on-Error)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut validator = JidokaIndexValidator::new(384); // 384-dim embeddings

    // Valid embeddings
    let valid_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    match validator.validate_embedding("doc1", &valid_embedding) {
        Ok(_) => println!("✅ doc1: Valid 384-dim embedding"),
        Err(e) => println!("❌ doc1: {}", e),
    }

    // Invalid: wrong dimensions
    let wrong_dims: Vec<f32> = vec![0.1, 0.2, 0.3];
    match validator.validate_embedding("doc2", &wrong_dims) {
        Ok(_) => println!("✅ doc2: Valid"),
        Err(e) => println!("❌ doc2: {} (Jidoka halt!)", e),
    }

    // Invalid: NaN values
    let nan_embedding: Vec<f32> = (0..384)
        .map(|i| if i == 100 { f32::NAN } else { 0.0 })
        .collect();
    match validator.validate_embedding("doc3", &nan_embedding) {
        Ok(_) => println!("✅ doc3: Valid"),
        Err(e) => println!("❌ doc3: {} (Jidoka halt!)", e),
    }

    let val_stats = validator.stats();
    println!("\n📊 Validation Statistics:");
    println!("  Total validations: {}", val_stats.total_validations);
    println!("  Successful: {}", val_stats.successful);
    println!("  Failed: {}", val_stats.failed);
    println!("  Halts triggered: {}\n", val_stats.halts);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. RAG ORACLE INTERFACE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let oracle = RagOracle::new();
    let oracle_stats = oracle.stats();

    println!("🔮 RAG Oracle Status:");
    println!("  Total documents: {}", oracle_stats.total_documents);
    println!("  Total chunks: {}", oracle_stats.total_chunks);
    println!("  Components: {}\n", oracle_stats.components);

    println!("💡 CLI Usage:");
    println!("  # Index stack documentation");
    println!("  batuta oracle --rag-index\n");
    println!("  # Query with RAG");
    println!("  batuta oracle --rag \"How do I train a model?\"\n");
    println!("  # TUI Dashboard");
    println!("  batuta oracle --rag-dashboard\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. TOYOTA WAY PRINCIPLES IN RAG");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🏭 Toyota Production System Applied:\n");
    println!("  ┌─────────────────┬────────────────────────────────────┐");
    println!("  │ Principle       │ RAG Implementation                 │");
    println!("  ├─────────────────┼────────────────────────────────────┤");
    println!("  │ Jidoka          │ Stop-on-error validation           │");
    println!("  │                 │ NaN/Inf detection, dim mismatch    │");
    println!("  ├─────────────────┼────────────────────────────────────┤");
    println!("  │ Poka-Yoke       │ Content hashing prevents stale     │");
    println!("  │                 │ indexes (BLAKE3 fingerprints)      │");
    println!("  ├─────────────────┼────────────────────────────────────┤");
    println!("  │ Heijunka        │ Load-leveled reindexing            │");
    println!("  │                 │ Priority queue by staleness        │");
    println!("  ├─────────────────┼────────────────────────────────────┤");
    println!("  │ Muda            │ Delta-only updates                 │");
    println!("  │                 │ Skip unchanged documents           │");
    println!("  ├─────────────────┼────────────────────────────────────┤");
    println!("  │ Kaizen          │ Continuous embedding improvement   │");
    println!("  │                 │ Model hash tracking                │");
    println!("  └─────────────────┴────────────────────────────────────┘\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. GROUND TRUTH CORPORA (Cross-Language)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("📚 Ground Truth Corpus Support:\n");
    println!("The RAG Oracle indexes both Rust and Python sources:");
    println!();

    // Demonstrate Python chunker configuration
    let python_chunker_config = ChunkerConfig::new(
        512,
        64,
        &[
            "\n## ",        // Markdown H2
            "\n### ",       // Markdown H3
            "\ndef ",       // Python function
            "\nclass ",     // Python class
            "\n    def ",   // Python method
            "\nasync def ", // Python async function
        ],
    );
    let _python_chunker = SemanticChunker::from_config(&python_chunker_config);

    println!("🐍 Python Chunking Delimiters:");
    println!("  - def     (function definitions)");
    println!("  - class   (class definitions)");
    println!("  -     def (method definitions)");
    println!("  - async def (async functions)");
    println!();

    // Sample Python document
    let python_doc = r#"
# HuggingFace Ground Truth Corpus

## Preprocessing Module

### Text Tokenization

```python
def preprocess_text(text: str) -> str:
    """Preprocess text for BERT tokenization.

    Args:
        text: Raw input text

    Returns:
        Cleaned and normalized text
    """
    text = text.strip().lower()
    text = ' '.join(text.split())
    return text
```

### Pipeline Creation

```python
def create_pipeline(task: str) -> Pipeline:
    """Create a HuggingFace inference pipeline.

    Args:
        task: Pipeline task (e.g., 'sentiment-analysis')

    Returns:
        Configured Pipeline instance
    """
    from transformers import pipeline
    return pipeline(task)
```

## Rust Equivalent (candle)

The equivalent Rust code using candle:

```rust
use candle_core::Tensor;
use candle_transformers::models::bert;

fn preprocess_text(text: &str) -> String {
    text.trim().to_lowercase().split_whitespace().collect()
}
```
"#;

    let python_chunks = chunker.split(python_doc);
    println!(
        "📄 Python doc chunked into {} semantic chunks:\n",
        python_chunks.len()
    );

    for (i, chunk) in python_chunks.iter().take(3).enumerate() {
        let preview: String = chunk.content.chars().take(50).collect();
        println!(
            "  Chunk {}: lines {}-{} ({} chars)",
            i + 1,
            chunk.start_line,
            chunk.end_line,
            chunk.content.len()
        );
        println!("    {}...\n", preview.replace('\n', " "));
    }

    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │            GROUND TRUTH CORPUS ARCHITECTURE                  │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │                                                             │");
    println!("  │  ┌──────────────────┐    ┌──────────────────┐             │");
    println!("  │  │  Rust Stack      │    │  Python Corpus   │             │");
    println!("  │  │  (trueno, etc)   │    │  (hf-gtc)        │             │");
    println!("  │  │  CLAUDE.md       │    │  CLAUDE.md       │             │");
    println!("  │  │  README.md       │    │  src/**/*.py     │             │");
    println!("  │  └────────┬─────────┘    └────────┬─────────┘             │");
    println!("  │           │                       │                        │");
    println!("  │           ▼                       ▼                        │");
    println!("  │  ┌─────────────────────────────────────────────────────┐  │");
    println!("  │  │              RAG Oracle Index (BM25)                 │  │");
    println!("  │  │  Cross-language search for ML patterns              │  │");
    println!("  │  └─────────────────────────────────────────────────────┘  │");
    println!("  │                         │                                  │");
    println!("  │                         ▼                                  │");
    println!("  │  Query: \"How do I tokenize text for BERT?\"              │");
    println!("  │         ↓                                                 │");
    println!("  │  Results: hf-gtc/preprocessing/tokenization.py           │");
    println!("  │           + candle/trueno Rust equivalent                │");
    println!("  │                                                             │");
    println!("  └─────────────────────────────────────────────────────────────┘\n");

    println!("💡 CLI Usage:");
    println!("  # Index stack docs AND ground truth corpora");
    println!("  batuta oracle --rag-index\n");
    println!("  # Query for Python ML patterns (cross-language)");
    println!("  batuta oracle --rag \"How do I tokenize text for BERT?\"\n");
    println!("  # Get Python recipe + Rust equivalent");
    println!("  batuta oracle --rag \"sentiment analysis pipeline\"\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("9. PRIVATE RAG CONFIGURATION (.batuta-private.toml)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🔒 Private repos can be indexed without committing paths to git.\n");
    println!("Create .batuta-private.toml at the project root (git-ignored):\n");
    println!("  [private]");
    println!("  rust_stack_dirs = [\"../rmedia\", \"../infra\", \"../assetgen\"]");
    println!("  rust_corpus_dirs = [\"../resolve-pipeline\"]");
    println!("  python_corpus_dirs = [\"../coursera-stats\", \"../interactive.paiml.com\"]\n");

    // Demonstrate PrivateConfig parsing
    use batuta::config::{PrivateConfig, PRIVATE_CONFIG_FILENAME};

    let toml_str = r#"
[private]
rust_stack_dirs = ["../rmedia", "../infra", "../assetgen"]
rust_corpus_dirs = ["../resolve-pipeline"]
python_corpus_dirs = ["../coursera-stats", "../interactive.paiml.com"]
"#;

    let private: PrivateConfig = toml::from_str(toml_str).unwrap();
    println!("📝 Parsed PrivateConfig:");
    println!("  Rust stack dirs: {:?}", private.private.rust_stack_dirs);
    println!("  Rust corpus dirs: {:?}", private.private.rust_corpus_dirs);
    println!(
        "  Python corpus dirs: {:?}",
        private.private.python_corpus_dirs
    );
    println!("  Total dirs: {}", private.dir_count());
    println!("  Config filename: {}\n", PRIVATE_CONFIG_FILENAME);

    // Demonstrate load_optional behavior
    println!("📂 Load Behavior:");
    println!("  Missing file  → Ok(None)  (silent, normal)");
    println!("  Malformed TOML → Err(...)  (warning printed, indexing continues)");
    println!("  Empty [private] → Ok(Some) with 0 dirs (no-op)\n");

    // Demonstrate empty config
    let empty: PrivateConfig = toml::from_str("[private]\n").unwrap();
    println!("  Empty config has_dirs: {}", empty.has_dirs());
    println!("  Empty config dir_count: {}\n", empty.dir_count());

    println!("💡 CLI Output (during batuta oracle --rag-index):");
    println!("  Private: 6 private directories merged from .batuta-private.toml\n");

    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │            PRIVATE RAG ARCHITECTURE                          │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │                                                             │");
    println!("  │  .batuta-private.toml ──▶ PrivateConfig::load_optional()   │");
    println!("  │  (git-ignored)              │                              │");
    println!("  │                             ▼                              │");
    println!("  │                    IndexConfig::apply_private()            │");
    println!("  │                    (merge into standard dirs)              │");
    println!("  │                             │                              │");
    println!("  │                             ▼                              │");
    println!("  │  ┌────────────────────────────────────────────────────┐   │");
    println!("  │  │              RAG Oracle Index (BM25)                │   │");
    println!("  │  │  Public stack + Private repos = unified search     │   │");
    println!("  │  └────────────────────────────────────────────────────┘   │");
    println!("  │                                                             │");
    println!("  └─────────────────────────────────────────────────────────────┘\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("10. MEDIA PRODUCTION (rmedia Oracle Integration)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🎬 rmedia is registered in the Oracle knowledge graph as Layer 7: Media.\n");

    // Demonstrate Oracle recommender for media queries
    use batuta::oracle::Recommender;

    let recommender = Recommender::new();

    // Query for media production
    let response = recommender.query("render video from MLT");
    println!("📊 Query: \"render video from MLT\"");
    println!("  Problem class: {}", response.problem_class);
    println!(
        "  Primary: {} ({}%)",
        response.primary.component,
        (response.primary.confidence * 100.0) as u32
    );
    for sup in &response.supporting {
        println!("  Supporting: {} ({}%)", sup.component, (sup.confidence * 100.0) as u32);
    }
    println!();

    // Show capabilities
    let caps = recommender.get_capabilities("rmedia");
    println!("🔧 rmedia capabilities ({} total):", caps.len());
    for cap in caps.iter().take(5) {
        println!("  - {}", cap);
    }
    println!("  ... and {} more\n", caps.len().saturating_sub(5));

    // Show integration
    if let Some(pattern) = recommender.get_integration("whisper-apr", "rmedia") {
        println!("🔗 Integration: whisper-apr → rmedia");
        println!("  Pattern: {}", pattern.pattern_name);
        println!("  {}\n", pattern.description);
    }

    println!("💡 CLI Usage:");
    println!("  batuta oracle --show rmedia");
    println!("  batuta oracle \"render video from MLT\"");
    println!("  batuta oracle --integrate whisper-apr,rmedia");
    println!("  batuta oracle --capabilities rmedia\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("11. INDEX PERSISTENCE (Section 9.7)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("📁 Persistent Storage:");
    println!("  Location: ~/.cache/batuta/rag/");
    println!("  Format:   JSON with BLAKE3 checksums (Jidoka)\n");

    println!("📄 Cache Files:");
    println!("  ├── manifest.json     # Version, checksums, timestamps");
    println!("  ├── index.json        # Inverted index (BM25 terms)");
    println!("  └── documents.json    # Document metadata + chunks\n");

    // Demonstrate persistence API
    use batuta::oracle::rag::persistence::{
        CorpusSource, PersistedDocuments, PersistedIndex, RagPersistence,
    };

    // Note: Using a temp dir for the demo to avoid modifying user's cache
    let temp_dir = std::env::temp_dir().join("rag_demo_cache");
    let persistence = RagPersistence::with_path(temp_dir.clone());

    // Create sample data
    let index = PersistedIndex {
        avg_doc_length: 89.4,
        ..Default::default()
    };
    let docs = PersistedDocuments {
        total_chunks: 142,
        ..Default::default()
    };
    let sources = vec![
        CorpusSource {
            id: "trueno".to_string(),
            commit: Some("abc123".to_string()),
            doc_count: 4,
            chunk_count: 42,
        },
        CorpusSource {
            id: "hf-ground-truth-corpus".to_string(),
            commit: Some("def456".to_string()),
            doc_count: 12,
            chunk_count: 100,
        },
    ];

    // Save index
    println!("💾 Save/Load Roundtrip Demo:\n");
    match persistence.save(&index, &docs, sources) {
        Ok(()) => println!("  ✓ Index saved to {:?}", temp_dir),
        Err(e) => println!("  ✗ Save failed: {}", e),
    }

    // Load and verify
    match persistence.load() {
        Ok(Some((loaded_index, loaded_docs, manifest))) => {
            println!("  ✓ Index loaded successfully");
            println!("    Version: {}", manifest.version);
            println!("    Sources: {} corpora", manifest.sources.len());
            println!("    Avg doc length: {:.1}", loaded_index.avg_doc_length);
            println!("    Total chunks: {}\n", loaded_docs.total_chunks);
        }
        Ok(None) => println!("  ⚠ No cached index found"),
        Err(e) => println!("  ✗ Load failed: {}", e),
    }

    // Get stats without full load
    match persistence.stats() {
        Ok(Some(manifest)) => {
            println!("📊 Quick Stats (manifest only):");
            println!("  Version: {}", manifest.version);
            println!("  Batuta version: {}", manifest.batuta_version);
            println!("  Indexed at: {} ms since epoch", manifest.indexed_at);
            for source in &manifest.sources {
                println!(
                    "  - {}: {} docs, {} chunks",
                    source.id, source.doc_count, source.chunk_count
                );
            }
            println!();
        }
        Ok(None) => println!("  No stats available"),
        Err(e) => println!("  Stats error: {}", e),
    }

    // Cleanup demo cache
    let _ = persistence.clear();
    let _ = std::fs::remove_dir(&temp_dir);

    println!("🔐 Integrity Validation (Jidoka):");
    println!("  - BLAKE3 checksums for index.json and documents.json");
    println!("  - Version compatibility check (major version match)");
    println!("  - Checksum mismatch triggers load failure (stop-on-error)");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │            PERSISTENCE ARCHITECTURE                          │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │                                                             │");
    println!("  │  Index (CLI)          Persist           Load (CLI)          │");
    println!("  │  ───────────          ───────           ──────────          │");
    println!("  │  batuta oracle        ┌───────┐         batuta oracle       │");
    println!("  │  --rag-index    ────▶ │ Cache │ ────▶   --rag \"query\"       │");
    println!("  │                       └───────┘                             │");
    println!("  │                           │                                 │");
    println!("  │                           ▼                                 │");
    println!("  │  batuta oracle   ──────▶ Stats                              │");
    println!("  │  --rag-stats            (no full load)                      │");
    println!("  │                                                             │");
    println!("  │  batuta oracle   ──────▶ Full Rebuild (two-phase save)      │");
    println!("  │  --rag-index-force                                          │");
    println!("  │                                                             │");
    println!("  └─────────────────────────────────────────────────────────────┘\n");

    println!("💡 CLI Usage:");
    println!("  # Index stack docs (saves to ~/.cache/batuta/rag/)");
    println!("  batuta oracle --rag-index\n");
    println!("  # Query (loads from cache automatically)");
    println!("  batuta oracle --rag \"How do I train a model?\"\n");
    println!("  # Show cache statistics");
    println!("  batuta oracle --rag-stats\n");
    println!("  # Force rebuild (old cache retained until save)");
    println!("  batuta oracle --rag-index-force\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("12. AUTO-UPDATE & FINGERPRINT CHANGE DETECTION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🔄 Three-Layer Freshness System:\n");
    println!("  The RAG index stays fresh via a layered auto-update system:");
    println!();

    println!("  Layer 1: Shell Auto-Fresh (ora-fresh)");
    println!("  ──────────────────────────────────────");
    println!("  On every shell login, ora-fresh runs in the background:");
    println!("  - Checks if ~/.cache/batuta/rag/.stale marker exists");
    println!("  - Checks if index is >24h old");
    println!("  - Triggers reindex only when needed");
    println!();

    println!("  Layer 2: Post-Commit Hooks (26 repos)");
    println!("  ──────────────────────────────────────");
    println!("  Every commit in any stack repo touches the stale marker:");
    println!("  - .git/hooks/post-commit: touch ~/.cache/batuta/rag/.stale");
    println!("  - Next ora-fresh picks this up and triggers reindex");
    println!("  - Zero overhead on commit (single touch call)");
    println!();

    println!("  Layer 3: Fingerprint-Based Change Detection (BLAKE3)");
    println!("  ────────────────────────────────────────────────────");
    println!("  On reindex, BLAKE3 fingerprints detect if anything changed:");
    println!("  - Content hash for every indexed file");
    println!("  - Chunker config hash (detect config changes)");
    println!("  - Model hash (detect embedding model changes)");
    println!("  - If nothing changed: skip entire reindex instantly");
    println!();

    // Demonstrate fingerprint-based change detection
    println!("📝 Fingerprint Change Detection Demo:\n");

    let doc_v1 = "fn hello() { println!(\"world\"); }";
    let doc_v2 = "fn hello() { println!(\"world!\"); }"; // One character change

    let fp_v1 = DocumentFingerprint::new(doc_v1.as_bytes(), &chunker_config, model_hash);
    let fp_v2 = DocumentFingerprint::new(doc_v2.as_bytes(), &chunker_config, model_hash);

    println!(
        "  File v1 hash: {:02x}{:02x}{:02x}{:02x}...",
        fp_v1.content_hash[0], fp_v1.content_hash[1], fp_v1.content_hash[2], fp_v1.content_hash[3]
    );
    println!(
        "  File v2 hash: {:02x}{:02x}{:02x}{:02x}... (one char changed)",
        fp_v2.content_hash[0], fp_v2.content_hash[1], fp_v2.content_hash[2], fp_v2.content_hash[3]
    );
    println!("  Needs reindex: {}", fp_v1.needs_reindex(&fp_v2));

    // Same content = same hash
    let fp_v1_again = DocumentFingerprint::new(doc_v1.as_bytes(), &chunker_config, model_hash);
    println!(
        "  Same content:  {} (no reindex needed)\n",
        !fp_v1.needs_reindex(&fp_v1_again)
    );

    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │            AUTO-UPDATE ARCHITECTURE                          │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │                                                             │");
    println!("  │  git commit ─────▶ post-commit hook                        │");
    println!("  │                    touch ~/.cache/batuta/rag/.stale         │");
    println!("  │                            │                                │");
    println!("  │                            ▼                                │");
    println!("  │  shell login ────▶ ora-fresh (background)                  │");
    println!("  │                    checks .stale marker + 24h age          │");
    println!("  │                            │                                │");
    println!("  │                            ▼                                │");
    println!("  │  batuta oracle ──▶ fingerprint check (BLAKE3)              │");
    println!("  │  --rag-index       compare content hashes                  │");
    println!("  │                    skip if nothing changed                  │");
    println!("  │                            │                                │");
    println!("  │                    (changed)│(unchanged)                    │");
    println!("  │                            │     └──▶ \"Index is current\"   │");
    println!("  │                            ▼                                │");
    println!("  │                    Full reindex (~30s)                      │");
    println!("  │                    Persist new fingerprints                 │");
    println!("  │                                                             │");
    println!("  └─────────────────────────────────────────────────────────────┘\n");

    println!("💡 CLI Usage:");
    println!("  # Check index freshness (runs automatically on shell login)");
    println!("  ora-fresh\n");
    println!("  # Index with fingerprint detection (skips if current)");
    println!("  batuta oracle --rag-index\n");
    println!("  # Force full reindex (ignores fingerprints)");
    println!("  batuta oracle --rag-index-force\n");

    println!("✅ RAG Oracle ready for production!");
    println!("   Run: batuta oracle --rag-index && batuta oracle --rag \"your query\"");
}
