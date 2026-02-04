//! PMAT Query integration for Oracle mode
//!
//! Provides function-level quality-annotated code search via `pmat query`,
//! optionally combined with RAG document-level retrieval for a hybrid view.

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

// ============================================================================
// Private helpers
// ============================================================================

/// Parse `pmat query` JSON output into structured results.
fn parse_pmat_query_output(json: &str) -> anyhow::Result<Vec<PmatQueryResult>> {
    let results: Vec<PmatQueryResult> = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("Failed to parse pmat query output: {e}"))?;
    Ok(results)
}

/// Invoke `pmat query` and return parsed results.
fn run_pmat_query(opts: &PmatQueryOptions) -> anyhow::Result<Vec<PmatQueryResult>> {
    let limit_str = opts.limit.to_string();
    let mut args: Vec<&str> = vec!["query", &opts.query, "--format", "json", "--limit", &limit_str];

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
    parse_pmat_query_output(&output)
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
    format!("{}{} {:.1}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty), score)
}

/// Format results as colored text.
fn pmat_format_results_text(query_text: &str, results: &[PmatQueryResult]) {
    println!("{}: {}", "PMAT Query".bright_cyan(), query_text);
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    for (i, r) in results.iter().enumerate() {
        let badge = grade_badge(&r.tdg_grade);
        let score_bar = tdg_score_bar(r.tdg_score, 10);
        println!(
            "{}. {} {}:{}  {}          {}",
            i + 1,
            badge,
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
        if let Some(ref src) = r.source {
            println!("   {}", "\u{2500}".repeat(40).dimmed());
            for line in src.lines().take(10) {
                println!("   {}", line.dimmed());
            }
            if src.lines().count() > 10 {
                println!("   {}", "...".dimmed());
            }
        }
        println!();
    }
}

/// Format results as JSON with query metadata envelope.
fn pmat_format_results_json(query_text: &str, results: &[PmatQueryResult]) -> anyhow::Result<()> {
    let json = serde_json::json!({
        "query": query_text,
        "source": "pmat",
        "result_count": results.len(),
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
            // Show source code for each result if available
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

/// Display combined PMAT + RAG results.
fn pmat_display_combined(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    rag_results: &[crate::oracle::rag::RetrievalResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::json!({
                "query": query_text,
                "pmat": {
                    "source": "pmat",
                    "result_count": pmat_results.len(),
                    "results": pmat_results,
                },
                "rag": {
                    "source": "rag",
                    "result_count": rag_results.len(),
                    "results": rag_results.iter().map(|r| {
                        serde_json::json!({
                            "component": r.component,
                            "source": r.source,
                            "score": r.score,
                            "content": r.content,
                        })
                    }).collect::<Vec<_>>(),
                }
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OracleOutputFormat::Markdown => {
            pmat_format_results_markdown(query_text, pmat_results);
            println!();
            println!("## RAG Document Results\n");
            println!("| # | Component | Source | Score |");
            println!("|---|-----------|--------|-------|");
            for (i, r) in rag_results.iter().enumerate() {
                println!("| {} | {} | {} | {:.3} |", i + 1, r.component, r.source, r.score);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{} + {}",
                "PMAT Functions".bright_cyan().bold(),
                "RAG Documents".bright_green().bold()
            );
            println!("{}", "\u{2500}".repeat(50).dimmed());
            println!();

            // PMAT section
            println!("{}", "Functions:".bright_cyan().bold());
            pmat_format_results_text(query_text, pmat_results);

            // RAG section
            println!("{}", "Documents:".bright_green().bold());
            use crate::oracle::rag::tui::inline;
            for (i, result) in rag_results.iter().enumerate() {
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
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            for r in pmat_results {
                if let Some(ref src) = r.source {
                    println!("// {}:{} - {}", r.file_path, r.start_line, r.function_name);
                    println!("{}", src);
                    println!();
                }
            }
            if pmat_results.iter().all(|r| r.source.is_none()) {
                eprintln!("No source code in results (try --pmat-include-source)");
                std::process::exit(1);
            }
        }
    }
    Ok(())
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
        "\"cache\" --rag  # combined function + document search".dimmed()
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

    let pmat_results = run_pmat_query(&opts)?;

    if pmat_results.is_empty() {
        println!(
            "{}",
            "No functions matched the query. Try broadening your search.".dimmed()
        );
        return Ok(());
    }

    if also_rag {
        // Combined mode: pmat + RAG
        let rag_results = match load_rag_results(&query_text)? {
            Some(results) => results,
            None => Vec::new(),
        };
        pmat_display_combined(&query_text, &pmat_results, &rag_results, format)?;
    } else {
        pmat_display_results(&query_text, &pmat_results, format)?;
    }

    Ok(())
}

/// Load RAG results for combined display, returning None (with warning) if index unavailable.
fn load_rag_results(
    query_text: &str,
) -> anyhow::Result<Option<Vec<crate::oracle::rag::RetrievalResult>>> {
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

    Ok(Some(results))
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
        // Badge should contain "A" (the ANSI codes are part of the string)
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
        // 10 filled blocks
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
        // Only required field is file_path and function_name; everything else has defaults
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
}
