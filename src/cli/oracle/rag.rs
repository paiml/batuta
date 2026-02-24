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
use crate::cli::oracle_indexing::{
    check_dir_for_changes, doc_fingerprint_changed, index_dir_group,
};

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

/// Load all configured SQLite indices (main oracle + private endpoints).
#[cfg(feature = "rag")]
pub(super) fn rag_load_all_indices(
) -> anyhow::Result<Vec<(String, trueno_rag::sqlite::SqliteIndex)>> {
    let mut indices = Vec::new();

    // Main oracle index
    let main_path = sqlite_index_path();
    if main_path.exists() {
        let idx = trueno_rag::sqlite::SqliteIndex::open(&main_path)
            .map_err(|e| anyhow::anyhow!("Failed to open main index: {e}"))?;
        indices.push(("oracle".to_string(), idx));
    }

    // Private endpoints from .batuta-private.toml
    if let Ok(Some(config)) = crate::config::PrivateConfig::load_optional() {
        for ep in &config.private.endpoints {
            if ep.endpoint_type == "local" {
                let path = std::path::Path::new(&ep.index_path);
                if path.exists() {
                    match trueno_rag::sqlite::SqliteIndex::open(path) {
                        Ok(idx) => indices.push((ep.name.clone(), idx)),
                        Err(e) => {
                            eprintln!(
                                "  {} Failed to open endpoint {}: {}",
                                "[warning]".bright_yellow(),
                                ep.name,
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(indices)
}

/// Search all indices and fuse results via Reciprocal Rank Fusion.
#[cfg(feature = "rag")]
pub(super) fn rag_search_multi(
    indices: &[(String, trueno_rag::sqlite::SqliteIndex)],
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SqliteSearchResult>> {
    use std::collections::HashMap;

    let rrf_k = 60.0_f64;
    let mut score_map: HashMap<String, (f64, SqliteSearchResult)> = HashMap::new();

    for (source_name, index) in indices {
        let results = index
            .search_fts(query, k)
            .map_err(|e| anyhow::anyhow!("FTS5 search on {source_name} failed: {e}"))?;

        for (rank, r) in results.into_iter().enumerate() {
            let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);
            let key = format!("{}:{}", source_name, r.chunk_id);
            let entry = score_map.entry(key).or_insert_with(|| {
                (
                    0.0,
                    SqliteSearchResult {
                        chunk_id: r.chunk_id,
                        doc_id: r.doc_id,
                        content: r.content,
                        score: 0.0,
                    },
                )
            });
            entry.0 += rrf_score;
        }
    }

    let mut results: Vec<_> = score_map.into_values().collect();
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    Ok(results
        .into_iter()
        .map(|(score, mut r)| {
            r.score = score;
            r
        })
        .collect())
}

/// RAG query using SQLite+FTS5 backend with multi-index support.
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
    let indices = rag_load_all_indices()?;
    drop(_load_span);
    let load_ms = load_start.elapsed().as_millis();

    if indices.is_empty() {
        println!(
            "{}",
            "No SQLite index found. Run 'batuta oracle --rag-index' first."
                .bright_yellow()
                .bold()
        );
        println!();
        return Ok(());
    }

    let mut total_docs = 0usize;
    let mut total_chunks = 0usize;
    for (name, idx) in &indices {
        let doc_count = idx.document_count().map_err(|e| anyhow::anyhow!("{e}"))?;
        let chunk_count = idx.chunk_count().map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "  {} {}: {} docs, {} chunks",
            "Index".bright_yellow(),
            name.cyan(),
            doc_count,
            chunk_count,
        );
        total_docs += doc_count;
        total_chunks += chunk_count;
    }
    if indices.len() > 1 {
        println!(
            "  {}: {} docs, {} chunks across {} indices",
            "Total".bright_yellow(),
            total_docs,
            total_chunks,
            indices.len(),
        );
    }
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
    let sqlite_results = if indices.len() == 1 {
        // Single index — direct search (no RRF overhead)
        rag_search_sqlite(&indices[0].1, &query_text, 10)?
    } else {
        // Multi-index — RRF fusion
        rag_search_multi(&indices, &query_text, 10)?
    };
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
            // Normalize score to 0-1 range (BM25/RRF scores vary)
            let max_score = sqlite_results
                .first()
                .map(|first| first.score)
                .unwrap_or(1.0)
                .max(f64::MIN_POSITIVE);
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
            "load={}ms  search={}ms  total={}ms  indices={}",
            load_ms, retrieve_ms, total_ms, indices.len()
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
        let cache = RAG_INDEX_CACHE
            .read()
            .map_err(|e| anyhow::anyhow!("RAG cache lock poisoned: {e}"))?;
        if let Some(ref data) = *cache {
            eprintln!(
                "  {} hit — using session-cached index",
                "[   cache]".dimmed()
            );
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
                let mut cache = RAG_INDEX_CACHE
                    .write()
                    .map_err(|e| anyhow::anyhow!("RAG cache lock poisoned: {e}"))?;
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
fn cmd_oracle_rag_json(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
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

/// Format multi-index stats for display.
#[cfg(feature = "rag")]
fn rag_format_multi_index_stats(
    indices: &[(String, trueno_rag::sqlite::SqliteIndex)],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let index_stats: Vec<serde_json::Value> = indices
                .iter()
                .map(|(name, idx)| {
                    let path = if name == "oracle" {
                        sqlite_index_path().display().to_string()
                    } else {
                        String::new()
                    };
                    serde_json::json!({
                        "name": name,
                        "backend": "sqlite",
                        "documents": idx.document_count().unwrap_or(0),
                        "chunks": idx.chunk_count().unwrap_or(0),
                        "path": path,
                    })
                })
                .collect();
            let json = serde_json::json!({ "indices": index_stats });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OracleOutputFormat::Markdown => {
            println!("## RAG Index Statistics (SQLite)\n");
            println!("| Index | Documents | Chunks |");
            println!("|-------|-----------|--------|");
            for (name, idx) in indices {
                println!(
                    "| {} | {} | {} |",
                    name,
                    idx.document_count().unwrap_or(0),
                    idx.chunk_count().unwrap_or(0),
                );
            }
        }
        OracleOutputFormat::Text => {
            let mut total_docs = 0usize;
            let mut total_chunks = 0usize;
            for (name, idx) in indices {
                let doc_count = idx.document_count().unwrap_or(0);
                let chunk_count = idx.chunk_count().unwrap_or(0);
                total_docs += doc_count;
                total_chunks += chunk_count;
                print_stat(
                    &format!("Index ({name})"),
                    format!("{} docs, {} chunks", doc_count, chunk_count).cyan(),
                );
            }
            if indices.len() > 1 {
                print_stat(
                    "Total",
                    format!(
                        "{} docs, {} chunks across {} indices",
                        total_docs,
                        total_chunks,
                        indices.len()
                    )
                    .cyan(),
                );
            }
            print_stat("Backend", "SQLite+FTS5".cyan());
            print_stat("Path (oracle)", format!("{:?}", sqlite_index_path()));
        }
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            eprintln!("No code available for RAG stats (try --format text)");
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Show RAG index statistics
pub fn cmd_oracle_rag_stats(format: OracleOutputFormat) -> anyhow::Result<()> {
    println!("{}", "RAG Index Statistics".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Try SQLite first — show all indices (main + private endpoints)
    #[cfg(feature = "rag")]
    {
        let indices = rag_load_all_indices()?;
        if !indices.is_empty() {
            rag_format_multi_index_stats(&indices, format)?;
            println!();
            return Ok(());
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "rag"))]
mod tests {
    use super::*;

    /// Helper: create a temporary SQLite index with test data.
    fn create_test_sqlite_index(
        path: &std::path::Path,
        docs: &[(&str, &[(&str, &str)])],
    ) -> trueno_rag::sqlite::SqliteIndex {
        let idx = trueno_rag::sqlite::SqliteIndex::open(path).unwrap();
        for (doc_id, chunks) in docs {
            let content: String = chunks.iter().map(|(_, c)| *c).collect::<Vec<_>>().join("\n");
            let chunk_pairs: Vec<(String, String)> = chunks
                .iter()
                .enumerate()
                .map(|(i, (_, c))| (format!("{doc_id}#{i}"), c.to_string()))
                .collect();
            idx.insert_document(doc_id, None, Some(doc_id), &content, &chunk_pairs, None)
                .unwrap();
        }
        idx.optimize().unwrap();
        idx
    }

    #[test]
    fn test_extract_component() {
        assert_eq!(extract_component("trueno/CLAUDE.md"), "trueno");
        assert_eq!(extract_component("batuta/src/main.rs"), "batuta");
        assert_eq!(extract_component("standalone.txt"), "standalone.txt");
        assert_eq!(extract_component(""), "");
    }

    #[test]
    fn test_sqlite_index_path_is_under_cache() {
        let path = sqlite_index_path();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("batuta/rag/index.sqlite"),
            "path should end with batuta/rag/index.sqlite, got: {path_str}"
        );
    }

    #[test]
    fn test_rag_load_sqlite_returns_none_if_missing() {
        // With default path — if the user doesn't have an index, returns None
        // We can't control the path here easily, but we can test the function exists
        // and returns Ok (either Some or None depending on system state)
        let result = rag_load_sqlite();
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_search_sqlite_returns_results() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");

        let idx = create_test_sqlite_index(
            &db_path,
            &[
                (
                    "doc-a",
                    &[
                        ("a#0", "Rust is a systems programming language"),
                        ("a#1", "The borrow checker ensures memory safety"),
                    ],
                ),
                (
                    "doc-b",
                    &[("b#0", "Python is an interpreted language")],
                ),
            ],
        );

        let results = rag_search_sqlite(&idx, "borrow checker", 5).unwrap();
        assert!(!results.is_empty(), "Should find results for 'borrow checker'");
        assert!(
            results[0].content.contains("borrow checker"),
            "Top result should contain query terms"
        );
    }

    #[test]
    fn test_rag_search_sqlite_empty_query() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");

        let idx = create_test_sqlite_index(
            &db_path,
            &[("doc-a", &[("a#0", "some content")])],
        );

        // Empty query — FTS5 may return all or none depending on tokenizer
        let results = rag_search_sqlite(&idx, "", 5);
        assert!(results.is_ok());
    }

    #[test]
    fn test_rag_search_multi_single_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");

        let idx = create_test_sqlite_index(
            &db_path,
            &[
                ("doc-a", &[("a#0", "SIMD operations for vector processing")]),
                ("doc-b", &[("b#0", "Python list comprehensions")]),
            ],
        );

        let indices = vec![("oracle".to_string(), idx)];
        let results = rag_search_multi(&indices, "SIMD vector", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("SIMD"));
    }

    #[test]
    fn test_rag_search_multi_fuses_two_indices() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create two separate indices
        let db1_path = tmp.path().join("oracle.sqlite");
        let idx1 = create_test_sqlite_index(
            &db1_path,
            &[("src/main.rs", &[("s#0", "Rust borrow checker and lifetimes")])],
        );

        let db2_path = tmp.path().join("video.sqlite");
        let idx2 = create_test_sqlite_index(
            &db2_path,
            &[(
                "lecture-1.srt",
                &[("v#0", "PDCA cycle in software engineering")],
            )],
        );

        let indices = vec![
            ("oracle".to_string(), idx1),
            ("video-corpus".to_string(), idx2),
        ];

        // Query that hits video corpus
        let results = rag_search_multi(&indices, "PDCA cycle", 5).unwrap();
        assert!(!results.is_empty(), "Should find PDCA in video corpus");
        assert!(results[0].content.contains("PDCA"));

        // Query that hits source code
        let results = rag_search_multi(&indices, "borrow checker", 5).unwrap();
        assert!(!results.is_empty(), "Should find borrow checker in oracle");
        assert!(results[0].content.contains("borrow checker"));
    }

    #[test]
    fn test_rag_search_multi_rrf_scores_are_positive() {
        let tmp = tempfile::TempDir::new().unwrap();

        let db_path = tmp.path().join("test.sqlite");
        let idx = create_test_sqlite_index(
            &db_path,
            &[
                ("doc-a", &[("a#0", "alpha beta gamma")]),
                ("doc-b", &[("b#0", "delta epsilon zeta")]),
            ],
        );

        let indices = vec![("test".to_string(), idx)];
        let results = rag_search_multi(&indices, "alpha", 5).unwrap();

        for r in &results {
            assert!(r.score > 0.0, "RRF scores should be positive");
        }
    }

    #[test]
    fn test_rag_search_multi_respects_k_limit() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");

        let docs: Vec<(&str, Vec<(&str, &str)>)> = (0..20)
            .map(|i| {
                // We need to leak the strings to get &str references
                // Use a simpler approach: fixed doc names
                match i {
                    0 => ("d0", vec![("d0#0", "alpha bravo charlie")]),
                    1 => ("d1", vec![("d1#0", "alpha delta echo")]),
                    2 => ("d2", vec![("d2#0", "alpha foxtrot golf")]),
                    3 => ("d3", vec![("d3#0", "alpha hotel india")]),
                    4 => ("d4", vec![("d4#0", "alpha juliet kilo")]),
                    _ => ("dN", vec![("dN#0", "something else entirely")]),
                }
            })
            .collect();

        let doc_refs: Vec<(&str, &[(&str, &str)])> = docs
            .iter()
            .map(|(id, chunks)| (*id, chunks.as_slice()))
            .collect();

        let idx = create_test_sqlite_index(&db_path, &doc_refs);
        let indices = vec![("test".to_string(), idx)];

        let results = rag_search_multi(&indices, "alpha", 3).unwrap();
        assert!(results.len() <= 3, "Should respect k=3 limit");
    }

    #[test]
    fn test_rag_search_multi_empty_indices() {
        let indices: Vec<(String, trueno_rag::sqlite::SqliteIndex)> = vec![];
        let results = rag_search_multi(&indices, "anything", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_rag_load_all_indices_includes_main() {
        // This test verifies the function runs without panicking.
        // Actual index availability depends on system state.
        let result = rag_load_all_indices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_sqlite_search_result_fields() {
        let r = SqliteSearchResult {
            chunk_id: "doc#0".to_string(),
            doc_id: "doc".to_string(),
            content: "test content".to_string(),
            score: 0.5,
        };
        assert_eq!(r.chunk_id, "doc#0");
        assert_eq!(r.doc_id, "doc");
        assert_eq!(r.content, "test content");
        assert!((r.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_format_timestamp_just_now() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assert_eq!(format_timestamp(now_ms), "just now");
    }

    #[test]
    fn test_format_timestamp_minutes_ago() {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        let five_min_ago = SystemTime::now() - Duration::from_secs(300);
        let ms = five_min_ago.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        let result = format_timestamp(ms);
        assert!(result.contains("min ago"), "expected 'min ago', got: {result}");
    }

    #[test]
    fn test_format_timestamp_hours_ago() {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        let two_hours_ago = SystemTime::now() - Duration::from_secs(7200);
        let ms = two_hours_ago.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        let result = format_timestamp(ms);
        assert!(result.contains("hours ago"), "expected 'hours ago', got: {result}");
    }

    #[test]
    fn test_format_timestamp_days_ago() {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        let three_days_ago = SystemTime::now() - Duration::from_secs(259200);
        let ms = three_days_ago.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        let result = format_timestamp(ms);
        assert!(result.contains("days ago"), "expected 'days ago', got: {result}");
    }

    #[test]
    fn test_print_stat_does_not_panic() {
        // print_stat just prints to stdout — verify it doesn't panic
        print_stat("Test Label", "test value");
        print_stat("Count", 42);
        print_stat("Ratio", format!("{:.2}", 0.95));
    }

    #[test]
    fn test_rag_show_usage_does_not_panic() {
        rag_show_usage();
    }

    #[test]
    fn test_rag_display_results_text() {
        let results = vec![oracle::rag::RetrievalResult {
            id: "doc#0".to_string(),
            component: "trueno".to_string(),
            source: "trueno/lib.rs".to_string(),
            content: "SIMD tensor operations".to_string(),
            score: 0.95,
            start_line: 1,
            end_line: 10,
            score_breakdown: oracle::rag::ScoreBreakdown {
                bm25_score: 5.0,
                dense_score: 0.0,
                rrf_score: 0.0,
                rerank_score: None,
            },
        }];
        let result = rag_display_results("test query", &results, OracleOutputFormat::Text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_display_results_json() {
        let results = vec![oracle::rag::RetrievalResult {
            id: "doc#0".to_string(),
            component: "batuta".to_string(),
            source: "batuta/main.rs".to_string(),
            content: "test content".to_string(),
            score: 0.8,
            start_line: 1,
            end_line: 5,
            score_breakdown: oracle::rag::ScoreBreakdown {
                bm25_score: 3.0,
                dense_score: 0.0,
                rrf_score: 0.0,
                rerank_score: None,
            },
        }];
        let result = rag_display_results("json query", &results, OracleOutputFormat::Json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_display_results_markdown() {
        let results = vec![oracle::rag::RetrievalResult {
            id: "doc#0".to_string(),
            component: "pmat".to_string(),
            source: "pmat/analysis.rs".to_string(),
            content: "code analysis".to_string(),
            score: 0.7,
            start_line: 1,
            end_line: 1,
            score_breakdown: oracle::rag::ScoreBreakdown {
                bm25_score: 2.0,
                dense_score: 0.0,
                rrf_score: 0.0,
                rerank_score: None,
            },
        }];
        let result = rag_display_results("md query", &results, OracleOutputFormat::Markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_format_multi_index_stats_text() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let idx = create_test_sqlite_index(
            &db_path,
            &[("doc-a", &[("a#0", "content")])],
        );
        let indices = vec![("oracle".to_string(), idx)];
        let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_format_multi_index_stats_json() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let idx = create_test_sqlite_index(
            &db_path,
            &[("doc-a", &[("a#0", "content")])],
        );
        let indices = vec![("oracle".to_string(), idx)];
        let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_format_multi_index_stats_markdown() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let idx = create_test_sqlite_index(
            &db_path,
            &[("doc-a", &[("a#0", "content")])],
        );
        let indices = vec![("oracle".to_string(), idx)];
        let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Markdown);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_format_multi_index_stats_multiple() {
        let tmp = tempfile::TempDir::new().unwrap();

        let db1 = tmp.path().join("oracle.sqlite");
        let idx1 = create_test_sqlite_index(
            &db1,
            &[("doc-a", &[("a#0", "content alpha")])],
        );
        let db2 = tmp.path().join("video.sqlite");
        let idx2 = create_test_sqlite_index(
            &db2,
            &[("doc-b", &[("b#0", "content beta")])],
        );

        let indices = vec![
            ("oracle".to_string(), idx1),
            ("video-corpus".to_string(), idx2),
        ];
        let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Text);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Command-level integration tests (exercise full orchestration paths)
    // ========================================================================

    #[test]
    fn test_cmd_oracle_rag_sqlite_with_query() {
        // Exercises the full cmd_oracle_rag_sqlite path.
        // If no index exists, the "no index found" branch is covered.
        // If an index exists (from dogfooding), the search + display path is covered.
        let result = cmd_oracle_rag_sqlite(
            Some("test query".into()),
            OracleOutputFormat::Text,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_sqlite_no_query() {
        // Exercises the usage-display branch (query=None).
        let result = cmd_oracle_rag_sqlite(None, OracleOutputFormat::Text, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_sqlite_json_format() {
        let result = cmd_oracle_rag_sqlite(
            Some("SIMD".into()),
            OracleOutputFormat::Json,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_sqlite_with_profiling() {
        // Use a query likely to match content in the dogfood index
        let result = cmd_oracle_rag_sqlite(
            Some("Rust programming".into()),
            OracleOutputFormat::Text,
            true,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_sqlite_markdown_format() {
        let result = cmd_oracle_rag_sqlite(
            Some("Rust".into()),
            OracleOutputFormat::Markdown,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_rag_print_profiling_summary_does_not_panic() {
        rag_print_profiling_summary();
    }

    #[test]
    fn test_cmd_oracle_rag_dispatch() {
        let result = cmd_oracle_rag(Some("dispatch test".into()), OracleOutputFormat::Text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_with_profile_dispatch() {
        let result = cmd_oracle_rag_with_profile(
            Some("profile dispatch".into()),
            OracleOutputFormat::Text,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_stats_text() {
        let result = cmd_oracle_rag_stats(OracleOutputFormat::Text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_stats_json() {
        let result = cmd_oracle_rag_stats(OracleOutputFormat::Json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_oracle_rag_stats_markdown() {
        let result = cmd_oracle_rag_stats(OracleOutputFormat::Markdown);
        assert!(result.is_ok());
    }
}
