//! RAG Oracle Demo - Retrieval-Augmented Generation for Stack Documentation
//!
//! Demonstrates the APR-Powered RAG Oracle with:
//! - Content-addressable indexing (BLAKE3)
//! - Hybrid retrieval (BM25 + dense)
//! - Heijunka load-leveled reindexing
//! - Jidoka stop-on-error validation
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

    println!("✅ RAG Oracle ready for production!");
    println!("   Run: batuta oracle --rag-index && batuta oracle --rag \"your query\"");
}
