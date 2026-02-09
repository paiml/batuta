//! RAG Oracle commands
//!
//! This module contains RAG (Retrieval Augmented Generation) related commands
//! for indexing and querying stack documentation.
//!
//! When the `rag` feature is enabled, queries are dispatched to SQLite+FTS5
//! via `trueno_rag::sqlite::SqliteIndex` for BM25-ranked results.
//! Otherwise falls back to in-memory `HybridRetriever` loaded from JSON.

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::types::OracleOutputFormat;
#[allow(unused_imports)]
use crate::cli::oracle_indexing::{check_dir_for_changes, doc_fingerprint_changed, index_dir_group};

#[cfg(not(feature = "rag"))]
use std::sync::{Arc, LazyLock, RwLock};

/// Session-scoped cache for loaded RAG index data (JSON fallback only).
/// Uses Arc to allow cheap cloning without copying the index.
/// When `rag` feature is enabled, SQLite handles caching natively.
#[cfg(not(feature = "rag"))]
static RAG_INDEX_CACHE: LazyLock<RwLock<Option<Arc<RagIndexData>>>> =
    LazyLock::new(|| RwLock::new(None));

// ============================================================================
// Helper Functions
// ============================================================================

/// Format a Unix timestamp (ms) as a human-readable string
fn format_timestamp(timestamp_ms: u64) -> String {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    let duration = Duration::from_millis(timestamp_ms);
    let datetime = UNIX_EPOCH + duration;

    // Calculate age
    let age = SystemTime::now()
        .duration_since(datetime)
        .unwrap_or(Duration::ZERO);

    if age.as_secs() < 60 {
        "just now".to_string()
    } else if age.as_secs() < 3600 {
        format!("{} min ago", age.as_secs() / 60)
    } else if age.as_secs() < 86400 {
        format!("{} hours ago", age.as_secs() / 3600)
    } else {
        format!("{} days ago", age.as_secs() / 86400)
    }
}

/// Print a labeled statistic with the label in bright yellow.
fn print_stat(label: &str, value: impl std::fmt::Display) {
    println!("{}: {}", label.bright_yellow(), value);
}

// ============================================================================
// SQLite query backend (rag feature)
// ============================================================================

/// A search result from the SQLite FTS5 backend.
#[cfg(feature = "rag")]
pub(super) struct SqliteSearchResult {
    /// Chunk identifier (e.g., "trueno/CLAUDE.md#5")
    pub(super) chunk_id: String,
    /// Document identifier (e.g., "trueno/CLAUDE.md")
    pub(super) doc_id: String,
    /// Chunk text content
    pub(super) content: String,
    /// BM25 relevance score (higher = more relevant)
    pub(super) score: f64,
}

/// Default SQLite database path for the RAG index.
#[cfg(feature = "rag")]
pub(super) fn sqlite_index_path() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("batuta/rag/index.sqlite")
}

/// Load RAG index from SQLite. Returns None if the database doesn't exist.
#[cfg(feature = "rag")]
pub(super) fn rag_load_sqlite() -> anyhow::Result<Option<trueno_rag::sqlite::SqliteIndex>> {
    let db_path = sqlite_index_path();
    if !db_path.exists() {
        return Ok(None);
    }

    let index = trueno_rag::sqlite::SqliteIndex::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to open SQLite index: {e}"))?;

    Ok(Some(index))
}

/// Search the SQLite FTS5 index.
#[cfg(feature = "rag")]
pub(super) fn rag_search_sqlite(
    index: &trueno_rag::sqlite::SqliteIndex,
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SqliteSearchResult>> {
    let fts_results = index
        .search_fts(query, k)
        .map_err(|e| anyhow::anyhow!("FTS5 search failed: {e}"))?;

    Ok(fts_results
        .into_iter()
        .map(|r| SqliteSearchResult {
            chunk_id: r.chunk_id,
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
        })
        .collect())
}

/// Extract component name from doc_id (e.g., "trueno/CLAUDE.md" → "trueno")
#[cfg(feature = "rag")]
pub(super) fn extract_component(doc_id: &str) -> String {
    doc_id.split('/').next().unwrap_or("unknown").to_string()
}

/// RAG query using SQLite+FTS5 backend.
#[cfg(feature = "rag")]
fn cmd_oracle_rag_sqlite(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;
    use std::time::Instant;

    let total_start = Instant::now();
    let _total_span = trace.then(|| span("total_query"));

    println!("{}", "RAG Oracle Mode".bright_cyan().bold());
    println!("{}", "(SQLite+FTS5 backend)".dimmed());
    if profile || trace {
        println!("{}", "(profiling enabled)".dimmed());
    }
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let load_start = Instant::now();
    let _load_span = trace.then(|| span("index_load"));
    let index = match rag_load_sqlite()? {
        Some(idx) => idx,
        None => {
            println!(
                "{}",
                "No SQLite index found. Run 'batuta oracle --rag-index' first."
                    .bright_yellow()
                    .bold()
            );
            println!();
            return Ok(());
        }
    };
    drop(_load_span);
    let load_ms = load_start.elapsed().as_millis();

    let doc_count = index
        .document_count()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let chunk_count = index.chunk_count().map_err(|e| anyhow::anyhow!("{e}"))?;

    println!(
        "{}: {} documents, {} chunks (SQLite)",
        "Index".bright_yellow(),
        doc_count,
        chunk_count
    );
    println!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let retrieve_start = Instant::now();
    let _retrieve_span = trace.then(|| span("fts5_search"));
    let sqlite_results = rag_search_sqlite(&index, &query_text, 10)?;
    drop(_retrieve_span);
    let retrieve_ms = retrieve_start.elapsed().as_millis();

    if sqlite_results.is_empty() {
        println!(
            "{}",
            "No results found. Try running --rag-index first.".dimmed()
        );
        return Ok(());
    }

    // Convert to RetrievalResult for display
    let results: Vec<oracle::rag::RetrievalResult> = sqlite_results
        .iter()
        .map(|r| {
            let component = extract_component(&r.doc_id);
            // Normalize score to 0-1 range (BM25 scores vary)
            let max_score = sqlite_results
                .first()
                .map(|first| first.score)
                .unwrap_or(1.0)
                .max(1.0);
            oracle::rag::RetrievalResult {
                id: r.chunk_id.clone(),
                component,
                source: r.doc_id.clone(),
                content: r.content.chars().take(200).collect(),
                score: r.score / max_score,
                start_line: 1,
                end_line: 1,
                score_breakdown: oracle::rag::ScoreBreakdown {
                    bm25_score: r.score,
                    dense_score: 0.0,
                    rrf_score: 0.0,
                    rerank_score: None,
                },
            }
        })
        .collect();

    rag_display_results(&query_text, &results, format)?;

    // Phase timing summary
    let total_ms = total_start.elapsed().as_millis();
    println!(
        "{}",
        format!(
            "load={}ms  search={}ms  total={}ms",
            load_ms, retrieve_ms, total_ms
        )
        .dimmed()
    );

    if profile || trace {
        rag_print_profiling_summary();
    }

    Ok(())
}

// ============================================================================
// JSON fallback query backend
// ============================================================================

/// Loaded RAG index data (JSON backend only; SQLite path doesn't use this).
#[cfg(not(feature = "rag"))]
pub(super) struct RagIndexData {
    pub(super) retriever: oracle::rag::HybridRetriever,
    pub(super) doc_count: usize,
    pub(super) chunk_count: usize,
    pub(super) chunk_contents: std::collections::HashMap<String, String>,
}

/// Try to load RAG index from JSON, returns None if not found.
/// Uses session-scoped cache to avoid re-loading from disk on every query.
#[cfg(not(feature = "rag"))]
pub(super) fn rag_load_index() -> anyhow::Result<Option<Arc<RagIndexData>>> {
    use oracle::rag::persistence::RagPersistence;

    // Check session cache first
    {
        let cache = RAG_INDEX_CACHE.read().unwrap();
        if let Some(ref data) = *cache {
            eprintln!("  {} hit — using session-cached index", "[   cache]".dimmed());
            println!(
                "{}: {} documents, {} chunks (session cache)",
                "Index".bright_green(),
                data.doc_count,
                data.chunk_count
            );
            println!();
            return Ok(Some(Arc::clone(data)));
        }
    }

    let persistence = RagPersistence::new();
    match persistence.load() {
        Ok(Some((persisted_index, persisted_docs, manifest))) => {
            let retriever = oracle::rag::HybridRetriever::from_persisted(persisted_index.clone());
            let doc_count = {
                let unique_docs: std::collections::HashSet<&str> = persisted_index
                    .doc_lengths
                    .keys()
                    .map(|k| k.split('#').next().unwrap_or(k.as_str()))
                    .collect();
                unique_docs.len()
            };
            let chunk_count = persisted_docs.total_chunks;
            let chunk_contents = persisted_docs.chunk_contents.clone();

            println!(
                "{}: Loaded from disk (indexed {})",
                "Index".bright_green(),
                format_timestamp(manifest.indexed_at)
            );
            println!(
                "  {} documents, {} chunks, {} sources",
                doc_count,
                chunk_count,
                manifest.sources.len()
            );
            println!();

            let data = Arc::new(RagIndexData {
                retriever,
                doc_count,
                chunk_count,
                chunk_contents,
            });

            // Store in session cache
            {
                let mut cache = RAG_INDEX_CACHE.write().unwrap();
                *cache = Some(Arc::clone(&data));
            }

            Ok(Some(data))
        }
        Ok(None) => {
            println!(
                "{}",
                "No cached index found. Run 'batuta oracle --rag-index' first."
                    .bright_yellow()
                    .bold()
            );
            println!();
            Ok(None)
        }
        Err(e) => {
            println!(
                "{}: {} - rebuilding index recommended",
                "Warning".bright_yellow(),
                e
            );
            println!();
            Ok(None)
        }
    }
}

// ============================================================================
// Display helpers
// ============================================================================

fn rag_format_results_json(
    query_text: &str,
    results: &[oracle::rag::RetrievalResult],
) -> anyhow::Result<()> {
    let json = serde_json::json!({
        "query": query_text,
        "results": results.iter().map(|r| {
            serde_json::json!({
                "component": r.component,
                "source": r.source,
                "score": r.score,
                "content": r.content,
            })
        }).collect::<Vec<_>>()
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn rag_format_results_markdown(query_text: &str, results: &[oracle::rag::RetrievalResult]) {
    println!("## RAG Query Results\n");
    println!("**Query:** {}\n", query_text);
    for (i, result) in results.iter().enumerate() {
        println!("### {}. {} ({})\n", i + 1, result.component, result.source);
        println!("**Score:** {:.3}\n", result.score);
        if !result.content.is_empty() {
            println!("```\n{}\n```\n", result.content);
        }
    }
}

fn rag_format_results_text(query_text: &str, results: &[oracle::rag::RetrievalResult]) {
    use oracle::rag::tui::inline;

    println!("{}: {}", "Query".bright_cyan(), query_text);
    println!();

    for (i, result) in results.iter().enumerate() {
        let score_bar = inline::score_bar(result.score, 10);
        println!(
            "{}. [{}] {} {}",
            i + 1,
            result.component.bright_yellow(),
            result.source.dimmed(),
            score_bar
        );
        if !result.content.is_empty() {
            let preview: String = result.content.chars().take(200).collect();
            println!("   {}", preview.dimmed());
        }
        println!();
    }
}

fn rag_show_usage() {
    println!(
        "{}",
        "Usage: batuta oracle --rag \"your query here\"".dimmed()
    );
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --rag".cyan(),
        "\"How do I train a model?\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --rag".cyan(),
        "\"SIMD tensor operations\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --rag-index".cyan(),
        "# Index stack documentation first".dimmed()
    );
}

/// Print profiling summary for RAG queries
fn rag_print_profiling_summary() {
    use oracle::rag::profiling::GLOBAL_METRICS;

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!("{}", "Profiling Summary".bright_cyan().bold());

    let summary = GLOBAL_METRICS.summary();
    for (name, stats) in &summary.spans {
        println!(
            "  {}: {:.2}ms (count: {})",
            name.bright_yellow(),
            stats.total_us as f64 / 1000.0,
            stats.count
        );
    }

    println!(
        "  {}: {:.1}%",
        "Cache hit rate".bright_yellow(),
        GLOBAL_METRICS.cache_hit_rate() * 100.0
    );
}

/// Format and display RAG results based on output format
fn rag_display_results(
    query_text: &str,
    results: &[oracle::rag::RetrievalResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => rag_format_results_json(query_text, results)?,
        OracleOutputFormat::Markdown => rag_format_results_markdown(query_text, results),
        OracleOutputFormat::Text => rag_format_results_text(query_text, results),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            eprintln!("No code available for RAG results (try --format text)");
            std::process::exit(1);
        }
    }
    Ok(())
}

// ============================================================================
// Public Commands
// ============================================================================

/// RAG-based query using indexed documentation (without profiling)
#[allow(dead_code, clippy::needless_return)]
pub fn cmd_oracle_rag(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
    // Dispatch to SQLite backend when available
    #[cfg(feature = "rag")]
    {
        return cmd_oracle_rag_sqlite(query, format, false, false);
    }

    #[cfg(not(feature = "rag"))]
    {
        cmd_oracle_rag_json(query, format)
    }
}

/// JSON fallback RAG query (when rag feature is not enabled)
#[cfg(not(feature = "rag"))]
fn cmd_oracle_rag_json(
    query: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::rag::DocumentIndex;

    println!("{}", "RAG Oracle Mode".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let index_data = match rag_load_index()? {
        Some(data) => data,
        None => return Ok(()),
    };

    println!(
        "{}: {} documents, {} chunks",
        "Index".bright_yellow(),
        index_data.doc_count,
        index_data.chunk_count
    );
    println!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(&query_text, &empty_index, 10);

    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content = snippet.chars().take(200).collect();
        }
    }

    if results.is_empty() {
        println!(
            "{}",
            "No results found. Try running --rag-index first.".dimmed()
        );
        return Ok(());
    }

    rag_display_results(&query_text, &results, format)?;
    Ok(())
}

/// RAG-based query with profiling support
#[allow(clippy::needless_return)]
pub fn cmd_oracle_rag_with_profile(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    // Dispatch to SQLite backend when available
    #[cfg(feature = "rag")]
    {
        return cmd_oracle_rag_sqlite(query, format, profile, trace);
    }

    #[cfg(not(feature = "rag"))]
    {
        cmd_oracle_rag_json_with_profile(query, format, profile, trace)
    }
}

/// JSON fallback RAG query with profiling (when rag feature is not enabled)
#[cfg(not(feature = "rag"))]
fn cmd_oracle_rag_json_with_profile(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;
    use oracle::rag::DocumentIndex;
    use std::time::Instant;

    let total_start = Instant::now();
    let _total_span = trace.then(|| span("total_query"));

    println!("{}", "RAG Oracle Mode".bright_cyan().bold());
    if profile || trace {
        println!("{}", "(profiling enabled)".dimmed());
    }
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let load_start = Instant::now();
    let _load_span = trace.then(|| span("index_load"));
    let index_data = match rag_load_index()? {
        Some(data) => data,
        None => return Ok(()),
    };
    drop(_load_span);
    let load_ms = load_start.elapsed().as_millis();

    println!(
        "{}: {} documents, {} chunks",
        "Index".bright_yellow(),
        index_data.doc_count,
        index_data.chunk_count
    );
    println!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let retrieve_start = Instant::now();
    let _retrieve_span = trace.then(|| span("retrieve"));
    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(&query_text, &empty_index, 10);
    drop(_retrieve_span);
    let retrieve_ms = retrieve_start.elapsed().as_millis();

    let enrich_start = Instant::now();
    let _enrich_span = trace.then(|| span("enrich_results"));
    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content = snippet.chars().take(200).collect();
        }
    }
    drop(_enrich_span);
    let enrich_ms = enrich_start.elapsed().as_millis();

    if results.is_empty() {
        println!(
            "{}",
            "No results found. Try running --rag-index first.".dimmed()
        );
        return Ok(());
    }

    rag_display_results(&query_text, &results, format)?;

    // Phase timing summary
    let total_ms = total_start.elapsed().as_millis();
    println!(
        "{}",
        format!(
            "load={}ms  retrieve={}ms  enrich={}ms  total={}ms",
            load_ms, retrieve_ms, enrich_ms, total_ms
        )
        .dimmed()
    );

    if profile || trace {
        rag_print_profiling_summary();
    }

    Ok(())
}

/// Show RAG index statistics
pub fn cmd_oracle_rag_stats(format: OracleOutputFormat) -> anyhow::Result<()> {
    println!("{}", "RAG Index Statistics".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Try SQLite first
    #[cfg(feature = "rag")]
    {
        let db_path = sqlite_index_path();
        if db_path.exists() {
            if let Ok(index) = trueno_rag::sqlite::SqliteIndex::open(&db_path) {
                let doc_count = index.document_count().unwrap_or(0);
                let chunk_count = index.chunk_count().unwrap_or(0);
                let db_size = std::fs::metadata(&db_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let indexed_at = index
                    .get_metadata("indexed_at")
                    .ok()
                    .flatten()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                let batuta_version = index
                    .get_metadata("batuta_version")
                    .ok()
                    .flatten()
                    .unwrap_or_default();

                match format {
                    OracleOutputFormat::Json => {
                        let json = serde_json::json!({
                            "backend": "sqlite",
                            "documents": doc_count,
                            "chunks": chunk_count,
                            "db_size_bytes": db_size,
                            "indexed_at": indexed_at,
                            "batuta_version": batuta_version,
                            "path": db_path.display().to_string(),
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    OracleOutputFormat::Markdown => {
                        println!("## RAG Index Statistics (SQLite)\n");
                        println!("| Property | Value |");
                        println!("|----------|-------|");
                        println!("| Backend | SQLite+FTS5 |");
                        println!("| Documents | {} |", doc_count);
                        println!("| Chunks | {} |", chunk_count);
                        println!("| DB Size | {:.1} MB |", db_size as f64 / 1_048_576.0);
                        println!("| Indexed | {} |", format_timestamp(indexed_at));
                        println!("| Batuta Version | {} |", batuta_version);
                    }
                    OracleOutputFormat::Text => {
                        print_stat("Backend", "SQLite+FTS5".cyan());
                        print_stat("Documents", doc_count.to_string().cyan());
                        print_stat("Chunks", chunk_count.to_string().cyan());
                        print_stat(
                            "DB Size",
                            format!("{:.1} MB", db_size as f64 / 1_048_576.0).cyan(),
                        );
                        print_stat("Indexed", format_timestamp(indexed_at).cyan());
                        print_stat("Batuta version", batuta_version.cyan());
                        print_stat("Path", format!("{:?}", db_path));
                    }
                    OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
                        eprintln!("No code available for RAG stats (try --format text)");
                        std::process::exit(1);
                    }
                }

                println!();
                return Ok(());
            }
        }
    }

    // Fallback: JSON persistence stats
    cmd_oracle_rag_stats_json(format)
}

/// Show RAG stats from JSON persistence (fallback)
fn cmd_oracle_rag_stats_json(format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::persistence::RagPersistence;

    let persistence = RagPersistence::new();

    match persistence.stats()? {
        Some(manifest) => {
            match format {
                OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
                    eprintln!("No code available for RAG stats (try --format text)");
                    std::process::exit(1);
                }
                OracleOutputFormat::Json => {
                    let json = serde_json::json!({
                        "backend": "json",
                        "version": manifest.version,
                        "indexed_at": manifest.indexed_at,
                        "batuta_version": manifest.batuta_version,
                        "sources": manifest.sources.iter().map(|s| {
                            serde_json::json!({
                                "id": s.id,
                                "commit": s.commit,
                                "doc_count": s.doc_count,
                                "chunk_count": s.chunk_count,
                            })
                        }).collect::<Vec<_>>()
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                OracleOutputFormat::Markdown => {
                    println!("## RAG Index Statistics (JSON)\n");
                    println!("| Property | Value |");
                    println!("|----------|-------|");
                    println!("| Backend | JSON |");
                    println!("| Version | {} |", manifest.version);
                    println!("| Batuta Version | {} |", manifest.batuta_version);
                    println!("| Indexed At | {} |", format_timestamp(manifest.indexed_at));
                    println!();
                    println!("### Sources\n");
                    for source in &manifest.sources {
                        println!(
                            "- **{}**: {} docs, {} chunks",
                            source.id, source.doc_count, source.chunk_count
                        );
                    }
                }
                OracleOutputFormat::Text => {
                    print_stat("Backend", "JSON".cyan());
                    print_stat("Index version", manifest.version.cyan());
                    print_stat("Batuta version", manifest.batuta_version.cyan());
                    print_stat("Indexed", format_timestamp(manifest.indexed_at).cyan());
                    print_stat("Cache path", format!("{:?}", persistence.cache_path()));
                    println!();

                    // Calculate totals
                    let total_docs: usize = manifest.sources.iter().map(|s| s.doc_count).sum();
                    let total_chunks: usize = manifest.sources.iter().map(|s| s.chunk_count).sum();

                    print_stat("Total", format!("{} documents", total_docs));
                    print_stat("Total", format!("{} chunks", total_chunks));
                    println!();

                    if !manifest.sources.is_empty() {
                        println!("{}", "Sources:".bright_yellow());
                        for source in &manifest.sources {
                            println!(
                                "  {} {}: {} docs, {} chunks",
                                "*".bright_blue(),
                                source.id.cyan(),
                                source.doc_count,
                                source.chunk_count
                            );
                            if let Some(commit) = &source.commit {
                                println!("    {} commit: {}", "".dimmed(), commit.dimmed());
                            }
                        }
                    }
                }
            }
        }
        None => {
            println!(
                "{}",
                "No cached index found. Run 'batuta oracle --rag-index' to create one."
                    .bright_yellow()
            );
        }
    }

    println!();
    Ok(())
}

pub fn cmd_oracle_rag_dashboard() -> anyhow::Result<()> {
    #[cfg(feature = "presentar-terminal")]
    {
        use oracle::rag::tui::OracleDashboard;
        let mut dashboard = OracleDashboard::new();
        dashboard.run()
    }
    #[cfg(not(feature = "presentar-terminal"))]
    {
        anyhow::bail!("TUI dashboard requires the 'tui' feature. Install with: cargo install batuta --features tui")
    }
}
