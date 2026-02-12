//! PMAT Query integration for Oracle mode
//!
//! Provides function-level quality-annotated code search via `pmat query`,
//! optionally combined with RAG document-level retrieval for a hybrid view.
//!
//! ## v2.0 Enhancements
//!
//! 1. RRF-fused ranking in combined mode (Cormack et al. 2009)
//! 2. Cross-project search via `--pmat-all-local`
//! 3. Result caching with mtime-based invalidation
//! 4. Quality distribution summary
//! 5. Documentation backlinks from RAG index

use crate::ansi_colors::Colorize;
use crate::tools;

use super::types::OracleOutputFormat;

// ============================================================================
// Types
// ============================================================================

/// A single result from `pmat query --format json`.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PmatQueryResult {
    pub file_path: String,
    pub function_name: String,
    #[serde(default)]
    pub signature: String,
    #[serde(default)]
    pub doc_comment: Option<String>,
    #[serde(default)]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: usize,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub tdg_score: f64,
    #[serde(default)]
    pub tdg_grade: String,
    #[serde(default)]
    pub complexity: u32,
    #[serde(default)]
    pub big_o: String,
    #[serde(default)]
    pub satd_count: u32,
    #[serde(default)]
    pub loc: usize,
    #[serde(default)]
    pub relevance_score: f64,
    #[serde(default)]
    pub source: Option<String>,
    /// Project name (set during cross-project search).
    #[serde(default)]
    pub project: Option<String>,
    /// RAG documentation backlinks found for this result's file.
    #[serde(default)]
    pub rag_backlinks: Vec<String>,
}

/// Options for invoking `pmat query`.
#[derive(Debug, Clone)]
pub struct PmatQueryOptions {
    pub query: String,
    pub project_path: Option<String>,
    pub limit: usize,
    pub min_grade: Option<String>,
    pub max_complexity: Option<u32>,
    pub include_source: bool,
}

/// Quality distribution summary computed from a set of results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QualitySummary {
    pub grades: std::collections::HashMap<String, usize>,
    pub avg_complexity: f64,
    pub total_satd: u32,
    pub complexity_range: (u32, u32),
}

/// A fused result that can be either a PMAT function hit or a RAG document hit.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum FusedResult {
    #[serde(rename = "function")]
    Function(Box<PmatQueryResult>),
    #[serde(rename = "document")]
    Document {
        component: String,
        source: String,
        score: f64,
        content: String,
    },
}

// ============================================================================
// Quality Summary
// ============================================================================

/// Compute quality distribution summary from results.
fn compute_quality_summary(results: &[PmatQueryResult]) -> QualitySummary {
    let mut grades: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_complexity: u64 = 0;
    let mut total_satd: u32 = 0;
    let mut min_cx = u32::MAX;
    let mut max_cx = 0u32;

    for r in results {
        *grades.entry(r.tdg_grade.clone()).or_insert(0) += 1;
        total_complexity += r.complexity as u64;
        total_satd += r.satd_count;
        min_cx = min_cx.min(r.complexity);
        max_cx = max_cx.max(r.complexity);
    }

    let avg_complexity = if results.is_empty() {
        0.0
    } else {
        total_complexity as f64 / results.len() as f64
    };

    if results.is_empty() {
        min_cx = 0;
    }

    QualitySummary {
        grades,
        avg_complexity,
        total_satd,
        complexity_range: (min_cx, max_cx),
    }
}

/// Format quality summary as a one-line string.
fn format_summary_line(summary: &QualitySummary) -> String {
    let mut grade_parts: Vec<String> = Vec::new();
    for grade in &["A", "B", "C", "D", "F"] {
        if let Some(&count) = summary.grades.get(*grade) {
            grade_parts.push(format!("{}{}", count, grade));
        }
    }
    let grades = if grade_parts.is_empty() {
        "none".to_string()
    } else {
        grade_parts.join(" ")
    };
    format!(
        "{} | Avg complexity: {:.1} | Total SATD: {} | Complexity: {}-{}",
        grades,
        summary.avg_complexity,
        summary.total_satd,
        summary.complexity_range.0,
        summary.complexity_range.1
    )
}

// ============================================================================
// RRF Fusion
// ============================================================================

/// Fuse PMAT and RAG results using Reciprocal Rank Fusion (k=60).
///
/// Returns interleaved results tagged by source type.
fn rrf_fuse_results(
    pmat_results: &[PmatQueryResult],
    rag_results: &[crate::oracle::rag::RetrievalResult],
    top_k: usize,
) -> Vec<(FusedResult, f64)> {
    let k = 60.0_f64;
    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut items: std::collections::HashMap<String, FusedResult> =
        std::collections::HashMap::new();

    // Accumulate PMAT ranked list
    for (rank, r) in pmat_results.iter().enumerate() {
        let id = format!("fn:{}:{}", r.file_path, r.function_name);
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        items
            .entry(id)
            .or_insert_with(|| FusedResult::Function(Box::new(r.clone())));
    }

    // Accumulate RAG ranked list
    for (rank, r) in rag_results.iter().enumerate() {
        let id = format!("doc:{}", r.id);
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        items.entry(id).or_insert_with(|| FusedResult::Document {
            component: r.component.clone(),
            source: r.source.clone(),
            score: r.score,
            content: r.content.clone(),
        });
    }

    // Normalize: max possible = 2 / (k + 1) if result is rank-1 in both lists
    let max_rrf = 2.0 / (k + 1.0);
    let mut fused: Vec<(FusedResult, f64)> = scores
        .into_iter()
        .filter_map(|(id, rrf_score)| {
            let normalized = (rrf_score / max_rrf).min(1.0);
            items.remove(&id).map(|item| (item, normalized))
        })
        .collect();

    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);
    fused
}

// ============================================================================
// Result Caching
// ============================================================================

/// Get the cache directory for pmat query results.
fn cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("batuta")
        .join("pmat-query")
}

/// Compute a cache key from query + project path.
fn cache_key(query: &str, project_path: Option<&str>) -> String {
    let input = format!("{}:{}", query, project_path.unwrap_or("."));
    // FNV-1a hash
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:016x}", hash)
}

/// Check if any .rs file in the project is newer than the cache file.
fn any_source_newer_than(
    project_path: &std::path::Path,
    cache_mtime: std::time::SystemTime,
) -> bool {
    let walker = match glob::glob(&format!("{}/**/*.rs", project_path.display())) {
        Ok(w) => w,
        Err(_) => return true,
    };
    for entry in walker.flatten() {
        if let Ok(meta) = entry.metadata() {
            if let Ok(mtime) = meta.modified() {
                if mtime > cache_mtime {
                    return true;
                }
            }
        }
    }
    false
}

/// Try to load cached results, returning None if cache miss or stale.
fn load_cached_results(query: &str, project_path: Option<&str>) -> Option<Vec<PmatQueryResult>> {
    let key = cache_key(query, project_path);
    let path = cache_dir().join(format!("{key}.json"));

    let cache_mtime = path.metadata().ok()?.modified().ok()?;

    let proj = std::path::Path::new(project_path.unwrap_or("."));
    if any_source_newer_than(proj, cache_mtime) {
        return None;
    }

    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save results to cache.
fn save_cache(query: &str, project_path: Option<&str>, results: &[PmatQueryResult]) {
    let dir = cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let key = cache_key(query, project_path);
    let path = dir.join(format!("{key}.json"));
    if let Ok(json) = serde_json::to_string(results) {
        let _ = std::fs::write(path, json);
    }
}

// ============================================================================
// RAG Backlinks
// ============================================================================

/// Attach RAG documentation backlinks to PMAT results by matching file paths.
fn attach_rag_backlinks(
    pmat_results: &mut [PmatQueryResult],
    chunk_contents: &std::collections::HashMap<String, String>,
) {
    for result in pmat_results.iter_mut() {
        // Match RAG chunk keys that contain the pmat result's file path
        // RAG keys look like: "component/src/foo.rs#chunk42"
        let file_suffix = &result.file_path;
        for key in chunk_contents.keys() {
            if key.contains(file_suffix) {
                result.rag_backlinks.push(key.clone());
            }
        }
        // Deduplicate and limit
        result.rag_backlinks.sort();
        result.rag_backlinks.dedup();
        result.rag_backlinks.truncate(3);
    }
}

// ============================================================================
// Cross-Project Search
// ============================================================================

/// Run pmat query across all discovered local PAIML projects, merging results.
/// Uses parallel execution via std::thread::scope for better performance.
fn run_cross_project_query(opts: &PmatQueryOptions) -> anyhow::Result<Vec<PmatQueryResult>> {
    use crate::oracle::LocalWorkspaceOracle;

    let mut oracle_ws = LocalWorkspaceOracle::new()?;
    oracle_ws.discover_projects()?;
    let projects: Vec<_> = oracle_ws.projects().iter().collect();

    eprintln!(
        "  {} Searching {} projects in parallel...",
        "[pmat-all]".dimmed(),
        projects.len()
    );

    // Parallel query across all projects
    let all_chunk_results: Vec<Vec<PmatQueryResult>> = std::thread::scope(|s| {
        let handles: Vec<_> = projects
            .iter()
            .map(|(name, project)| {
                let project_opts = PmatQueryOptions {
                    query: opts.query.clone(),
                    project_path: Some(project.path.to_string_lossy().to_string()),
                    limit: opts.limit,
                    min_grade: opts.min_grade.clone(),
                    max_complexity: opts.max_complexity,
                    include_source: opts.include_source,
                };
                let name = (*name).clone();
                s.spawn(move || match run_pmat_query(&project_opts) {
                    Ok(mut results) => {
                        for r in &mut results {
                            r.project = Some(name.clone());
                            r.file_path = format!("{}/{}", name, r.file_path);
                        }
                        results
                    }
                    Err(_) => Vec::new(),
                })
            })
            .collect();

        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    // Merge all results
    let mut all_results: Vec<PmatQueryResult> = all_chunk_results.into_iter().flatten().collect();

    // Sort by relevance descending, truncate to limit
    all_results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(opts.limit);

    Ok(all_results)
}

// ============================================================================
// Core helpers
// ============================================================================

/// Parse `pmat query` JSON output into structured results.
fn parse_pmat_query_output(json: &str) -> anyhow::Result<Vec<PmatQueryResult>> {
    let results: Vec<PmatQueryResult> = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("Failed to parse pmat query output: {e}"))?;
    Ok(results)
}

/// Invoke `pmat query` and return parsed results.
fn run_pmat_query(opts: &PmatQueryOptions) -> anyhow::Result<Vec<PmatQueryResult>> {
    // Check cache first
    if let Some(cached) = load_cached_results(&opts.query, opts.project_path.as_deref()) {
        eprintln!("  {} hit — using cached results", "[   cache]".dimmed());
        return Ok(cached);
    }

    let limit_str = opts.limit.to_string();
    let mut args: Vec<&str> = vec![
        "query",
        &opts.query,
        "--format",
        "json",
        "--limit",
        &limit_str,
    ];

    let grade_val;
    if let Some(ref grade) = opts.min_grade {
        grade_val = grade.clone();
        args.push("--min-grade");
        args.push(&grade_val);
    }

    let complexity_str;
    if let Some(max) = opts.max_complexity {
        complexity_str = max.to_string();
        args.push("--max-complexity");
        args.push(&complexity_str);
    }

    if opts.include_source {
        args.push("--include-source");
    }

    let working_dir = opts.project_path.as_ref().map(std::path::Path::new);
    let output = tools::run_tool("pmat", &args, working_dir)?;
    let results = parse_pmat_query_output(&output)?;

    // Save to cache
    save_cache(&opts.query, opts.project_path.as_deref(), &results);

    Ok(results)
}

/// Grade badge with color.
fn grade_badge(grade: &str) -> String {
    match grade {
        "A" => format!("[{}]", "A".bright_green().bold()),
        "B" => format!("[{}]", "B".green()),
        "C" => format!("[{}]", "C".bright_yellow()),
        "D" => format!("[{}]", "D".yellow()),
        "F" => format!("[{}]", "F".bright_red().bold()),
        other => format!("[{}]", other.dimmed()),
    }
}

/// Score bar for TDG score (0-100 scale).
fn tdg_score_bar(score: f64, width: usize) -> String {
    let filled = ((score / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{} {:.1}",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(empty),
        score
    )
}

/// Print quality summary line in text mode.
fn print_quality_summary(results: &[PmatQueryResult]) {
    if results.is_empty() {
        return;
    }
    let summary = compute_quality_summary(results);
    println!(
        "{}: {}",
        "Summary".bright_yellow().bold(),
        format_summary_line(&summary)
    );
    println!();
}

/// Format results as colored text.
fn pmat_format_results_text(query_text: &str, results: &[PmatQueryResult]) {
    println!("{}: {}", "PMAT Query".bright_cyan(), query_text);
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    for (i, r) in results.iter().enumerate() {
        let badge = grade_badge(&r.tdg_grade);
        let score_bar = tdg_score_bar(r.tdg_score, 10);
        let project_prefix = r
            .project
            .as_ref()
            .map(|p| format!("[{}] ", p.bright_blue()))
            .unwrap_or_default();
        println!(
            "{}. {} {}{}:{}  {}          {}",
            i + 1,
            badge,
            project_prefix,
            r.file_path.cyan(),
            r.start_line,
            r.function_name.bright_yellow(),
            score_bar
        );
        if !r.signature.is_empty() {
            println!("   {}", r.signature.dimmed());
        }
        println!(
            "   Complexity: {} | Big-O: {} | SATD: {}",
            r.complexity, r.big_o, r.satd_count
        );
        if let Some(ref doc) = r.doc_comment {
            let preview: String = doc.chars().take(120).collect();
            println!("   {}", preview.dimmed());
        }
        if !r.rag_backlinks.is_empty() {
            println!(
                "   {} {}",
                "See also:".bright_green(),
                r.rag_backlinks.join(", ").dimmed()
            );
        }
        if let Some(ref src) = r.source {
            println!("   {}", "\u{2500}".repeat(40).dimmed());
            for line in src.lines().take(10) {
                #[cfg(feature = "syntect")]
                crate::cli::syntax::print_highlighted_line(
                    line,
                    crate::cli::syntax::Language::Rust,
                    "   ",
                );
                #[cfg(not(feature = "syntect"))]
                println!("   {}", line);
            }
            if src.lines().count() > 10 {
                println!("   {}", "...".dimmed());
            }
        }
        println!();
    }

    print_quality_summary(results);
}

/// Format results as JSON with query metadata envelope.
fn pmat_format_results_json(query_text: &str, results: &[PmatQueryResult]) -> anyhow::Result<()> {
    let summary = compute_quality_summary(results);
    let json = serde_json::json!({
        "query": query_text,
        "source": "pmat",
        "result_count": results.len(),
        "summary": {
            "grades": summary.grades,
            "avg_complexity": summary.avg_complexity,
            "total_satd": summary.total_satd,
            "complexity_range": [summary.complexity_range.0, summary.complexity_range.1],
        },
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

/// Format results as a markdown table.
fn pmat_format_results_markdown(query_text: &str, results: &[PmatQueryResult]) {
    println!("## PMAT Query Results\n");
    println!("**Query:** {}\n", query_text);
    println!("| # | Grade | File | Function | TDG | Complexity | Big-O |");
    println!("|---|-------|------|----------|-----|------------|-------|");
    for (i, r) in results.iter().enumerate() {
        println!(
            "| {} | {} | {}:{} | `{}` | {:.1} | {} | {} |",
            i + 1,
            r.tdg_grade,
            r.file_path,
            r.start_line,
            r.function_name,
            r.tdg_score,
            r.complexity,
            r.big_o
        );
    }
    let summary = compute_quality_summary(results);
    println!("\n**Summary:** {}", format_summary_line(&summary));
}

/// Display results in the requested format.
fn pmat_display_results(
    query_text: &str,
    results: &[PmatQueryResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => pmat_format_results_json(query_text, results)?,
        OracleOutputFormat::Markdown => pmat_format_results_markdown(query_text, results),
        OracleOutputFormat::Text => pmat_format_results_text(query_text, results),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            for r in results {
                if let Some(ref src) = r.source {
                    println!("// {}:{} - {}", r.file_path, r.start_line, r.function_name);
                    println!("{}", src);
                    println!();
                }
            }
            if results.iter().all(|r| r.source.is_none()) {
                eprintln!("No source code in results (try --pmat-include-source)");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

/// Display RRF-fused combined PMAT + RAG results.
fn pmat_display_combined(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    rag_results: &[crate::oracle::rag::RetrievalResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let fused = rrf_fuse_results(pmat_results, rag_results, 20);

    match format {
        OracleOutputFormat::Json => display_combined_json(query_text, pmat_results, &fused)?,
        OracleOutputFormat::Markdown => display_combined_markdown(query_text, pmat_results, &fused),
        OracleOutputFormat::Text => display_combined_text(pmat_results, &fused),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => display_combined_code(&fused),
    }
    Ok(())
}

fn display_combined_json(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    fused: &[(FusedResult, f64)],
) -> anyhow::Result<()> {
    let summary = compute_quality_summary(pmat_results);
    let json = serde_json::json!({
        "query": query_text,
        "mode": "rrf_fused",
        "k": 60,
        "result_count": fused.len(),
        "summary": {
            "grades": summary.grades,
            "avg_complexity": summary.avg_complexity,
            "total_satd": summary.total_satd,
            "complexity_range": [summary.complexity_range.0, summary.complexity_range.1],
        },
        "results": fused.iter().map(|(item, score)| {
            let mut v = serde_json::to_value(item).unwrap_or_default();
            if let Some(obj) = v.as_object_mut() {
                obj.insert("rrf_score".to_string(), serde_json::json!(score));
            }
            v
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn display_combined_markdown(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    fused: &[(FusedResult, f64)],
) {
    println!("## Combined PMAT + RAG Results (RRF-fused)\n");
    println!("**Query:** {}\n", query_text);
    println!("| # | Type | Source | Score |");
    println!("|---|------|--------|-------|");
    for (i, (item, score)) in fused.iter().enumerate() {
        match item {
            FusedResult::Function(r) => {
                println!(
                    "| {} | fn | {}:{} `{}` [{}] | {:.3} |",
                    i + 1,
                    r.file_path,
                    r.start_line,
                    r.function_name,
                    r.tdg_grade,
                    score
                );
            }
            FusedResult::Document {
                component, source, ..
            } => {
                println!(
                    "| {} | doc | [{}] {} | {:.3} |",
                    i + 1,
                    component,
                    source,
                    score
                );
            }
        }
    }
    let summary = compute_quality_summary(pmat_results);
    println!(
        "\n**Summary (functions):** {}",
        format_summary_line(&summary)
    );
}

fn display_combined_text(pmat_results: &[PmatQueryResult], fused: &[(FusedResult, f64)]) {
    println!("{} (RRF k=60)", "Combined Search".bright_cyan().bold());
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    for (i, (item, score)) in fused.iter().enumerate() {
        display_combined_text_item(i, item, *score);
    }

    print_quality_summary(pmat_results);
}

fn display_combined_text_item(i: usize, item: &FusedResult, score: f64) {
    let score_pct = (score * 100.0) as usize;
    let bar_filled = (score * 10.0).round() as usize;
    let bar_empty = 10_usize.saturating_sub(bar_filled);
    let bar = format!(
        "{}{} {:3}%",
        "\u{2588}".repeat(bar_filled),
        "\u{2591}".repeat(bar_empty),
        score_pct,
    );

    match item {
        FusedResult::Function(r) => {
            let badge = grade_badge(&r.tdg_grade);
            let project_prefix = r
                .project
                .as_ref()
                .map(|p| format!("[{}] ", p.bright_blue()))
                .unwrap_or_default();
            println!(
                "{}. {} {} {}{}:{}  {}  {}",
                i + 1,
                "[fn]".bright_cyan(),
                badge,
                project_prefix,
                r.file_path.cyan(),
                r.start_line,
                r.function_name.bright_yellow(),
                bar,
            );
            println!(
                "   Complexity: {} | Big-O: {} | SATD: {}",
                r.complexity, r.big_o, r.satd_count
            );
            if !r.rag_backlinks.is_empty() {
                println!(
                    "   {} {}",
                    "See also:".bright_green(),
                    r.rag_backlinks.join(", ").dimmed()
                );
            }
        }
        FusedResult::Document {
            component,
            source,
            content,
            ..
        } => {
            println!(
                "{}. {} [{}] {} {}",
                i + 1,
                "[doc]".bright_green(),
                component.bright_yellow(),
                source.dimmed(),
                bar,
            );
            if !content.is_empty() {
                let preview: String = content.chars().take(200).collect();
                println!("   {}", preview.dimmed());
            }
        }
    }
    println!();
}

fn display_combined_code(fused: &[(FusedResult, f64)]) {
    for (item, _) in fused {
        if let FusedResult::Function(r) = item {
            if let Some(ref src) = r.source {
                println!("// {}:{} - {}", r.file_path, r.start_line, r.function_name);
                println!("{}", src);
                println!();
            }
        }
    }
    let has_source = fused
        .iter()
        .any(|(item, _)| matches!(item, FusedResult::Function(r) if r.source.is_some()));
    if !has_source {
        eprintln!("No source code in results (try --pmat-include-source)");
        std::process::exit(1);
    }
}

/// Show usage hint when no query is provided.
fn show_pmat_query_usage() {
    println!(
        "{}",
        "Usage: batuta oracle --pmat-query \"your query here\"".dimmed()
    );
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"error handling\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"serialize\" --pmat-min-grade A".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"allocator\" --pmat-include-source --format json".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"cache\" --rag  # combined RRF-fused search".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"tokenizer\" --pmat-all-local  # search all projects".dimmed()
    );
}

// ============================================================================
// Public command entry point
// ============================================================================

/// Execute PMAT query, optionally combined with RAG retrieval.
#[allow(clippy::too_many_arguments)]
pub fn cmd_oracle_pmat_query(
    query: Option<String>,
    project_path: Option<String>,
    limit: usize,
    min_grade: Option<String>,
    max_complexity: Option<u32>,
    include_source: bool,
    also_rag: bool,
    all_local: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    // Check pmat availability
    let registry = tools::ToolRegistry::detect();
    if registry.pmat.is_none() {
        eprintln!(
            "{}: pmat is not installed or not in PATH.",
            "Error".bright_red().bold()
        );
        eprintln!();
        eprintln!("Install pmat:");
        eprintln!("  cargo install pmat");
        eprintln!();
        eprintln!("Or check: https://github.com/paiml/pmat");
        return Ok(());
    }

    let query_text = match query {
        Some(q) if !q.is_empty() => q,
        _ => {
            show_pmat_query_usage();
            return Ok(());
        }
    };

    println!("{}", "PMAT Query Mode".bright_cyan().bold());
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    let opts = PmatQueryOptions {
        query: query_text.clone(),
        project_path,
        limit,
        min_grade,
        max_complexity,
        include_source,
    };

    let mut pmat_results = if all_local {
        println!(
            "{}",
            "Cross-project search (all local PAIML projects)".dimmed()
        );
        println!();
        run_cross_project_query(&opts)?
    } else {
        run_pmat_query(&opts)?
    };

    if pmat_results.is_empty() {
        println!(
            "{}",
            "No functions matched the query. Try broadening your search.".dimmed()
        );
        return Ok(());
    }

    if also_rag {
        // Combined mode: pmat + RAG with RRF fusion
        let rag_results = match load_rag_results(&query_text)? {
            Some((results, chunk_contents)) => {
                // Attach backlinks from RAG index to pmat results
                attach_rag_backlinks(&mut pmat_results, &chunk_contents);
                results
            }
            None => Vec::new(),
        };
        pmat_display_combined(&query_text, &pmat_results, &rag_results, format)?;
    } else {
        pmat_display_results(&query_text, &pmat_results, format)?;
    }

    Ok(())
}

/// Load RAG results and chunk contents for combined display.
/// Returns None (with warning) if index unavailable.
///
/// When the `rag` feature is enabled, uses SQLite+FTS5 backend directly.
/// Otherwise falls back to JSON-based `HybridRetriever`.
#[cfg(feature = "rag")]
fn load_rag_results(
    query_text: &str,
) -> anyhow::Result<
    Option<(
        Vec<crate::oracle::rag::RetrievalResult>,
        std::collections::HashMap<String, String>,
    )>,
> {
    let index = match super::rag::rag_load_sqlite()? {
        Some(idx) => idx,
        None => {
            eprintln!(
                "{}: RAG index not available. Showing PMAT results only.",
                "Warning".bright_yellow()
            );
            eprintln!("  Run 'batuta oracle --rag-index' to enable combined search.");
            eprintln!();
            return Ok(None);
        }
    };

    let sqlite_results = super::rag::rag_search_sqlite(&index, query_text, 10)?;

    let results: Vec<crate::oracle::rag::RetrievalResult> = sqlite_results
        .iter()
        .map(|r| {
            let component = super::rag::extract_component(&r.doc_id);
            let max_score = sqlite_results
                .first()
                .map(|first| first.score)
                .unwrap_or(1.0)
                .max(1.0);
            crate::oracle::rag::RetrievalResult {
                id: r.chunk_id.clone(),
                component,
                source: r.doc_id.clone(),
                content: r.content.chars().take(200).collect(),
                score: r.score / max_score,
                start_line: 1,
                end_line: 1,
                score_breakdown: crate::oracle::rag::ScoreBreakdown {
                    bm25_score: r.score,
                    dense_score: 0.0,
                    rrf_score: 0.0,
                    rerank_score: None,
                },
            }
        })
        .collect();

    // SQLite stores content in the DB, so chunk_contents HashMap is empty.
    // Backlinks will be a no-op (acceptable tradeoff vs cloning 389K strings).
    Ok(Some((results, std::collections::HashMap::new())))
}

/// Load RAG results and chunk contents for combined display (JSON fallback).
/// Returns None (with warning) if index unavailable.
#[cfg(not(feature = "rag"))]
fn load_rag_results(
    query_text: &str,
) -> anyhow::Result<
    Option<(
        Vec<crate::oracle::rag::RetrievalResult>,
        std::collections::HashMap<String, String>,
    )>,
> {
    use crate::oracle::rag::DocumentIndex;

    let index_data = match super::rag::rag_load_index()? {
        Some(data) => data,
        None => {
            eprintln!(
                "{}: RAG index not available. Showing PMAT results only.",
                "Warning".bright_yellow()
            );
            eprintln!("  Run 'batuta oracle --rag-index' to enable combined search.");
            eprintln!();
            return Ok(None);
        }
    };

    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(query_text, &empty_index, 10);

    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content.clone_from(snippet);
        }
    }

    Ok(Some((results, index_data.chunk_contents.clone())))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"[
            {
                "file_path": "src/pipeline.rs",
                "function_name": "validate_stage",
                "signature": "fn validate_stage(&self, stage: &Stage) -> Result<()>",
                "doc_comment": "Validates a pipeline stage.",
                "start_line": 142,
                "end_line": 185,
                "language": "rust",
                "tdg_score": 92.5,
                "tdg_grade": "A",
                "complexity": 4,
                "big_o": "O(n)",
                "satd_count": 0,
                "loc": 43,
                "relevance_score": 0.87,
                "source": null
            }
        ]"#
    }

    #[test]
    fn test_parse_valid_json() {
        let results = parse_pmat_query_output(sample_json()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].function_name, "validate_stage");
        assert_eq!(results[0].file_path, "src/pipeline.rs");
        assert_eq!(results[0].tdg_grade, "A");
        assert!((results[0].tdg_score - 92.5).abs() < f64::EPSILON);
        assert_eq!(results[0].complexity, 4);
        assert_eq!(results[0].big_o, "O(n)");
        assert_eq!(results[0].start_line, 142);
        assert_eq!(results[0].end_line, 185);
    }

    #[test]
    fn test_parse_null_doc_comment() {
        let json = r#"[{
            "file_path": "src/lib.rs",
            "function_name": "foo",
            "signature": "fn foo()",
            "doc_comment": null,
            "start_line": 1,
            "end_line": 5,
            "language": "rust",
            "tdg_score": 50.0,
            "tdg_grade": "C",
            "complexity": 2,
            "big_o": "O(1)",
            "satd_count": 0,
            "loc": 4,
            "relevance_score": 0.5,
            "source": null
        }]"#;
        let results = parse_pmat_query_output(json).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].doc_comment.is_none());
        assert!(results[0].source.is_none());
    }

    #[test]
    fn test_parse_with_source() {
        let json = r#"[{
            "file_path": "src/main.rs",
            "function_name": "main",
            "signature": "fn main()",
            "doc_comment": null,
            "start_line": 1,
            "end_line": 3,
            "language": "rust",
            "tdg_score": 100.0,
            "tdg_grade": "A",
            "complexity": 1,
            "big_o": "O(1)",
            "satd_count": 0,
            "loc": 3,
            "relevance_score": 1.0,
            "source": "fn main() {\n    println!(\"hello\");\n}"
        }]"#;
        let results = parse_pmat_query_output(json).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].source.is_some());
        assert!(results[0].source.as_ref().unwrap().contains("println!"));
    }

    #[test]
    fn test_parse_empty_array() {
        let results = parse_pmat_query_output("[]").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_pmat_query_output("not json");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to parse pmat query output"));
    }

    #[test]
    fn test_parse_multiple_results() {
        let json = r#"[
            {
                "file_path": "a.rs",
                "function_name": "alpha",
                "tdg_score": 90.0,
                "tdg_grade": "A",
                "complexity": 2,
                "big_o": "O(1)",
                "satd_count": 0,
                "relevance_score": 0.9
            },
            {
                "file_path": "b.rs",
                "function_name": "beta",
                "tdg_score": 60.0,
                "tdg_grade": "C",
                "complexity": 12,
                "big_o": "O(n^2)",
                "satd_count": 3,
                "relevance_score": 0.6
            }
        ]"#;
        let results = parse_pmat_query_output(json).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].function_name, "alpha");
        assert_eq!(results[1].function_name, "beta");
        assert_eq!(results[1].complexity, 12);
        assert_eq!(results[1].satd_count, 3);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let result = PmatQueryResult {
            file_path: "src/lib.rs".to_string(),
            function_name: "process".to_string(),
            signature: "fn process(data: &[u8]) -> Vec<u8>".to_string(),
            doc_comment: Some("Process raw bytes.".to_string()),
            start_line: 10,
            end_line: 25,
            language: "rust".to_string(),
            tdg_score: 85.0,
            tdg_grade: "B".to_string(),
            complexity: 6,
            big_o: "O(n)".to_string(),
            satd_count: 1,
            loc: 15,
            relevance_score: 0.75,
            source: None,
            project: None,
            rag_backlinks: Vec::new(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: PmatQueryResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.function_name, "process");
        assert!((deserialized.tdg_score - 85.0).abs() < f64::EPSILON);
        assert_eq!(deserialized.tdg_grade, "B");
    }

    #[test]
    fn test_grade_badge_a() {
        let badge = grade_badge("A");
        assert!(badge.contains('A'));
        assert!(badge.starts_with('['));
        assert!(badge.ends_with(']'));
    }

    #[test]
    fn test_grade_badge_f() {
        let badge = grade_badge("F");
        assert!(badge.contains('F'));
    }

    #[test]
    fn test_grade_badge_unknown() {
        let badge = grade_badge("X");
        assert!(badge.contains('X'));
    }

    #[test]
    fn test_tdg_score_bar_full() {
        let bar = tdg_score_bar(100.0, 10);
        assert!(bar.contains("100.0"));
        assert_eq!(bar.matches('\u{2588}').count(), 10);
        assert_eq!(bar.matches('\u{2591}').count(), 0);
    }

    #[test]
    fn test_tdg_score_bar_zero() {
        let bar = tdg_score_bar(0.0, 10);
        assert!(bar.contains("0.0"));
        assert_eq!(bar.matches('\u{2588}').count(), 0);
        assert_eq!(bar.matches('\u{2591}').count(), 10);
    }

    #[test]
    fn test_tdg_score_bar_half() {
        let bar = tdg_score_bar(50.0, 10);
        assert!(bar.contains("50.0"));
        assert_eq!(bar.matches('\u{2588}').count(), 5);
        assert_eq!(bar.matches('\u{2591}').count(), 5);
    }

    #[test]
    fn test_pmat_query_options_construction() {
        let opts = PmatQueryOptions {
            query: "error".to_string(),
            project_path: Some("/home/user/project".to_string()),
            limit: 5,
            min_grade: Some("B".to_string()),
            max_complexity: Some(10),
            include_source: true,
        };
        assert_eq!(opts.query, "error");
        assert_eq!(opts.limit, 5);
        assert_eq!(opts.min_grade.as_deref(), Some("B"));
        assert_eq!(opts.max_complexity, Some(10));
        assert!(opts.include_source);
    }

    #[test]
    fn test_parse_minimal_fields() {
        let json = r#"[{
            "file_path": "x.rs",
            "function_name": "f"
        }]"#;
        let results = parse_pmat_query_output(json).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "x.rs");
        assert_eq!(results[0].function_name, "f");
        assert!(results[0].signature.is_empty());
        assert_eq!(results[0].tdg_score, 0.0);
        assert_eq!(results[0].complexity, 0);
    }

    // ========================================================================
    // v2.0 tests
    // ========================================================================

    #[test]
    fn test_quality_summary_single() {
        let results = vec![PmatQueryResult {
            file_path: "a.rs".into(),
            function_name: "f".into(),
            tdg_grade: "A".into(),
            complexity: 5,
            satd_count: 1,
            ..Default::default()
        }];
        let s = compute_quality_summary(&results);
        assert_eq!(s.grades.get("A"), Some(&1));
        assert!((s.avg_complexity - 5.0).abs() < f64::EPSILON);
        assert_eq!(s.total_satd, 1);
        assert_eq!(s.complexity_range, (5, 5));
    }

    #[test]
    fn test_quality_summary_multiple() {
        let results = vec![
            PmatQueryResult {
                file_path: "a.rs".into(),
                function_name: "f".into(),
                tdg_grade: "A".into(),
                complexity: 2,
                satd_count: 0,
                ..Default::default()
            },
            PmatQueryResult {
                file_path: "b.rs".into(),
                function_name: "g".into(),
                tdg_grade: "B".into(),
                complexity: 8,
                satd_count: 3,
                ..Default::default()
            },
            PmatQueryResult {
                file_path: "c.rs".into(),
                function_name: "h".into(),
                tdg_grade: "A".into(),
                complexity: 4,
                satd_count: 0,
                ..Default::default()
            },
        ];
        let s = compute_quality_summary(&results);
        assert_eq!(s.grades.get("A"), Some(&2));
        assert_eq!(s.grades.get("B"), Some(&1));
        assert!((s.avg_complexity - 14.0 / 3.0).abs() < 0.01);
        assert_eq!(s.total_satd, 3);
        assert_eq!(s.complexity_range, (2, 8));
    }

    #[test]
    fn test_quality_summary_empty() {
        let s = compute_quality_summary(&[]);
        assert!(s.grades.is_empty());
        assert!((s.avg_complexity).abs() < f64::EPSILON);
        assert_eq!(s.total_satd, 0);
    }

    #[test]
    fn test_format_summary_line() {
        let mut grades = std::collections::HashMap::new();
        grades.insert("A".to_string(), 3);
        grades.insert("B".to_string(), 1);
        let s = QualitySummary {
            grades,
            avg_complexity: 5.5,
            total_satd: 2,
            complexity_range: (1, 12),
        };
        let line = format_summary_line(&s);
        assert!(line.contains("3A"));
        assert!(line.contains("1B"));
        assert!(line.contains("5.5"));
        assert!(line.contains("SATD: 2"));
        assert!(line.contains("1-12"));
    }

    #[test]
    fn test_rrf_fuse_pmat_only() {
        let pmat = vec![PmatQueryResult {
            file_path: "a.rs".into(),
            function_name: "f".into(),
            relevance_score: 0.9,
            tdg_grade: "A".into(),
            ..Default::default()
        }];
        let fused = rrf_fuse_results(&pmat, &[], 10);
        assert_eq!(fused.len(), 1);
        assert!(fused[0].1 > 0.0);
        assert!(matches!(fused[0].0, FusedResult::Function(_)));
    }

    #[test]
    fn test_rrf_fuse_both() {
        use crate::oracle::rag::RetrievalResult;
        use crate::oracle::rag::ScoreBreakdown;

        let pmat = vec![PmatQueryResult {
            file_path: "a.rs".into(),
            function_name: "f".into(),
            relevance_score: 0.9,
            tdg_grade: "A".into(),
            ..Default::default()
        }];
        let rag = vec![RetrievalResult {
            id: "doc1".into(),
            component: "trueno".into(),
            source: "trueno/README.md".into(),
            content: "some content".into(),
            score: 0.8,
            start_line: 1,
            end_line: 10,
            score_breakdown: ScoreBreakdown::default(),
        }];
        let fused = rrf_fuse_results(&pmat, &rag, 10);
        assert_eq!(fused.len(), 2);
        // Both should have positive scores
        assert!(fused[0].1 > 0.0);
        assert!(fused[1].1 > 0.0);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = cache_key("error handling", Some("/home/user/project"));
        let k2 = cache_key("error handling", Some("/home/user/project"));
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different() {
        let k1 = cache_key("error", Some("/a"));
        let k2 = cache_key("serialize", Some("/a"));
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_attach_rag_backlinks() {
        let mut results = vec![PmatQueryResult {
            file_path: "src/pipeline.rs".into(),
            function_name: "f".into(),
            ..Default::default()
        }];
        let mut chunks = std::collections::HashMap::new();
        chunks.insert(
            "batuta/src/pipeline.rs#42".to_string(),
            "chunk content".to_string(),
        );
        chunks.insert(
            "trueno/src/simd.rs#1".to_string(),
            "other content".to_string(),
        );

        attach_rag_backlinks(&mut results, &chunks);
        assert_eq!(results[0].rag_backlinks.len(), 1);
        assert!(results[0].rag_backlinks[0].contains("pipeline.rs"));
    }

    #[test]
    fn test_attach_rag_backlinks_no_match() {
        let mut results = vec![PmatQueryResult {
            file_path: "src/unique_file.rs".into(),
            function_name: "f".into(),
            ..Default::default()
        }];
        let mut chunks = std::collections::HashMap::new();
        chunks.insert("trueno/src/simd.rs#1".to_string(), "content".to_string());

        attach_rag_backlinks(&mut results, &chunks);
        assert!(results[0].rag_backlinks.is_empty());
    }

    #[test]
    fn test_fused_result_serialization() {
        let fused = FusedResult::Function(Box::new(PmatQueryResult {
            file_path: "a.rs".into(),
            function_name: "f".into(),
            tdg_grade: "A".into(),
            ..Default::default()
        }));
        let json = serde_json::to_string(&fused).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("a.rs"));
    }

    #[test]
    fn test_fused_result_document_serialization() {
        let fused = FusedResult::Document {
            component: "trueno".into(),
            source: "trueno/README.md".into(),
            score: 0.85,
            content: "content".into(),
        };
        let json = serde_json::to_string(&fused).unwrap();
        assert!(json.contains("\"type\":\"document\""));
        assert!(json.contains("trueno"));
    }
}

// ============================================================================
// Default impl for test convenience
// ============================================================================

impl Default for PmatQueryResult {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            function_name: String::new(),
            signature: String::new(),
            doc_comment: None,
            start_line: 0,
            end_line: 0,
            language: String::new(),
            tdg_score: 0.0,
            tdg_grade: String::new(),
            complexity: 0,
            big_o: String::new(),
            satd_count: 0,
            loc: 0,
            relevance_score: 0.0,
            source: None,
            project: None,
            rag_backlinks: Vec::new(),
        }
    }
}
