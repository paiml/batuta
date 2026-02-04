//! RAG Oracle commands
//!
//! This module contains RAG (Retrieval Augmented Generation) related commands
//! for indexing and querying stack documentation.

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::types::OracleOutputFormat;
use crate::cli::oracle_indexing::{check_dir_for_changes, doc_fingerprint_changed, index_dir_group};

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

/// Loaded RAG index data
pub(super) struct RagIndexData {
    pub(super) retriever: oracle::rag::HybridRetriever,
    pub(super) doc_count: usize,
    pub(super) chunk_count: usize,
    pub(super) chunk_contents: std::collections::HashMap<String, String>,
}

/// Try to load RAG index, returns None if not found
pub(super) fn rag_load_index() -> anyhow::Result<Option<RagIndexData>> {
    use oracle::rag::persistence::RagPersistence;

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
                "{}: Loaded from cache (indexed {})",
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

            Ok(Some(RagIndexData {
                retriever,
                doc_count,
                chunk_count,
                chunk_contents,
            }))
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
#[allow(dead_code)]
pub fn cmd_oracle_rag(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
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
            result.content.clone_from(snippet);
        }
    }

    if results.is_empty() {
        println!(
            "{}",
            "No results found. Try running --rag-index first.".dimmed()
        );
        return Ok(());
    }

    match format {
        OracleOutputFormat::Json => rag_format_results_json(&query_text, &results)?,
        OracleOutputFormat::Markdown => rag_format_results_markdown(&query_text, &results),
        OracleOutputFormat::Text => rag_format_results_text(&query_text, &results),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            eprintln!("No code available for RAG results (try --format text)");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// RAG-based query with profiling support
pub fn cmd_oracle_rag_with_profile(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;
    use oracle::rag::DocumentIndex;

    let _total_span = trace.then(|| span("total_query"));

    println!("{}", "RAG Oracle Mode".bright_cyan().bold());
    if profile || trace {
        println!("{}", "(profiling enabled)".dimmed());
    }
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let _load_span = trace.then(|| span("index_load"));
    let index_data = match rag_load_index()? {
        Some(data) => data,
        None => return Ok(()),
    };
    drop(_load_span);

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

    let _retrieve_span = trace.then(|| span("retrieve"));
    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(&query_text, &empty_index, 10);
    drop(_retrieve_span);

    let _enrich_span = trace.then(|| span("enrich_results"));
    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content.clone_from(snippet);
        }
    }
    drop(_enrich_span);

    if results.is_empty() {
        println!(
            "{}",
            "No results found. Try running --rag-index first.".dimmed()
        );
        return Ok(());
    }

    rag_display_results(&query_text, &results, format)?;

    if profile || trace {
        rag_print_profiling_summary();
    }

    Ok(())
}

/// Show RAG index statistics
pub fn cmd_oracle_rag_stats(format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::persistence::RagPersistence;

    println!("{}", "RAG Index Statistics".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

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
                    println!("## RAG Index Statistics\n");
                    println!("| Property | Value |");
                    println!("|----------|-------|");
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
    use oracle::rag::tui::OracleDashboard;

    let mut dashboard = OracleDashboard::new();
    dashboard.run()
}
