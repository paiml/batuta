//! Oracle command implementations
//!
//! This module contains all Oracle-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::oracle_indexing::{check_dir_for_changes, doc_fingerprint_changed, index_dir_group};

/// Oracle output format
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OracleOutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
    /// Raw code output only — no metadata, no colors
    Code,
}

// ============================================================================
// RAG Oracle Commands
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
struct RagIndexData {
    retriever: oracle::rag::HybridRetriever,
    doc_count: usize,
    chunk_count: usize,
    chunk_contents: std::collections::HashMap<String, String>,
}

/// Try to load RAG index, returns None if not found
fn rag_load_index() -> anyhow::Result<Option<RagIndexData>> {
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

/// RAG-based query using indexed documentation
pub fn cmd_oracle_rag(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::DocumentIndex;

    println!("{}", "🔍 RAG Oracle Mode".bright_cyan().bold());
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
        OracleOutputFormat::Code => {
            eprintln!("No code available for RAG results (try --format text)");
            std::process::exit(1);
        }
    }

    Ok(())
}

// ============================================================================
// Local Workspace Oracle Commands
// ============================================================================

fn local_print_summary(summary: &oracle::local_workspace::WorkspaceSummary) {
    println!(
        "{}: {} PAIML projects discovered in ~/src",
        "Summary".bright_yellow(),
        summary.total_projects
    );
    println!(
        "  {} {} (use crates.io versions for deps)",
        summary.projects_with_changes.to_string().bright_red(),
        "dirty".bright_red()
    );
    println!(
        "  {} {} (ready to push)",
        summary.projects_with_unpushed.to_string().bright_yellow(),
        "unpushed".bright_yellow()
    );
    println!(
        "  {} {} (using local versions)",
        (summary.total_projects - summary.projects_with_changes - summary.projects_with_unpushed)
            .to_string()
            .bright_green(),
        "clean".bright_green()
    );
    println!();
}

fn local_show_dirty_summary(projects: &[&oracle::local_workspace::LocalProject]) {
    use oracle::local_workspace::DevState;

    let dirty: Vec<_> = projects
        .iter()
        .filter(|p| p.dev_state == DevState::Dirty)
        .collect();

    println!(
        "{} {} projects with uncommitted changes:",
        "🔴".bright_red(),
        dirty.len()
    );
    println!();
    for project in &dirty {
        println!(
            "  {} {} ({} files)",
            "●".bright_red(),
            project.name.bright_white().bold(),
            project.git_status.modified_count
        );
    }
    println!();
    println!(
        "{}",
        "These projects are in active development - stack uses crates.io versions for deps"
            .dimmed()
    );
}

fn local_show_project_details(project: &oracle::local_workspace::LocalProject) {
    use oracle::local_workspace::DevState;

    let (status_icon, state_label) = match project.dev_state {
        DevState::Dirty => ("●".bright_red(), "DIRTY".bright_red()),
        DevState::Unpushed => ("◐".bright_yellow(), "UNPUSHED".bright_yellow()),
        DevState::Clean => ("○".bright_green(), "clean".bright_green()),
    };

    let version_info = match &project.published_version {
        Some(pub_v) if pub_v == &project.local_version => format!("v{}", project.local_version)
            .bright_green()
            .to_string(),
        Some(pub_v) => format!("v{} → v{}", pub_v, project.local_version)
            .bright_yellow()
            .to_string(),
        None => format!("v{} (unpublished)", project.local_version)
            .dimmed()
            .to_string(),
    };

    println!(
        "  {} {} {} [{}] {}",
        status_icon,
        project.name.bright_white().bold(),
        version_info,
        project.git_status.branch.dimmed(),
        state_label
    );

    if project.git_status.has_changes {
        println!(
            "      {} modified files",
            project.git_status.modified_count.to_string().bright_red()
        );
    }
    if project.git_status.unpushed_commits > 0 {
        println!(
            "      {} unpushed commits",
            project
                .git_status
                .unpushed_commits
                .to_string()
                .bright_yellow()
        );
    }
    if !project.paiml_dependencies.is_empty() {
        let deps: Vec<_> = project
            .paiml_dependencies
            .iter()
            .map(|d| {
                if d.is_path_dep {
                    format!("{}(path)", d.name)
                } else {
                    format!("{}@{}", d.name, d.required_version)
                }
            })
            .collect();
        println!("      deps: {}", deps.join(", ").dimmed());
    }
}

fn local_show_drift(drifts: &[oracle::local_workspace::VersionDrift]) {
    use oracle::local_workspace::DriftType;

    if drifts.is_empty() {
        return;
    }

    println!("{}", "📊 Version Drift".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());

    for drift in drifts {
        let icon = match drift.drift_type {
            DriftType::LocalAhead => "↑".bright_green(),
            DriftType::LocalBehind => "↓".bright_red(),
            DriftType::NotPublished => "○".dimmed(),
            DriftType::InSync => "✓".bright_green(),
        };
        let msg = match drift.drift_type {
            DriftType::LocalAhead => "ready to publish",
            DriftType::LocalBehind => "needs update",
            DriftType::NotPublished => "not published",
            DriftType::InSync => "in sync",
        };
        println!(
            "  {} {} {} → {} ({})",
            icon,
            drift.name.bright_white(),
            drift.published_version.dimmed(),
            drift.local_version.bright_yellow(),
            msg.dimmed()
        );
    }
    println!();
}

fn local_show_publish_order_text(order: &oracle::local_workspace::PublishOrder) {
    println!("{}", "🚀 Suggested Publish Order".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    if !order.cycles.is_empty() {
        println!("{} Dependency cycles detected:", "⚠️".bright_yellow());
        for cycle in &order.cycles {
            println!("  {}", cycle.join(" → ").bright_red());
        }
        println!();
    }

    let needs_publish: Vec<_> = order.order.iter().filter(|s| s.needs_publish).collect();

    if needs_publish.is_empty() {
        println!(
            "{}",
            "✅ All projects are up to date with crates.io".bright_green()
        );
    } else {
        println!(
            "Crates to publish ({} total):",
            needs_publish.len().to_string().bright_yellow()
        );
        println!();

        for (i, step) in needs_publish.iter().enumerate() {
            let blocked = if step.blocked_by.is_empty() {
                "ready".bright_green().to_string()
            } else {
                format!("after: {}", step.blocked_by.join(", "))
                    .dimmed()
                    .to_string()
            };

            println!(
                "  {}. {} v{} ({})",
                (i + 1).to_string().bright_cyan(),
                step.name.bright_white().bold(),
                step.version.bright_yellow(),
                blocked
            );
        }
    }
    println!();
}

fn local_show_usage() {
    println!("{}", "Usage:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --local".cyan(),
        "# Show all local PAIML projects".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --publish-order".cyan(),
        "# Show suggested publish order".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --local --publish-order".cyan(),
        "# Show both".dimmed()
    );
}

/// Local workspace discovery and multi-project intelligence
pub fn cmd_oracle_local(
    show_status: bool,
    show_dirty: bool,
    show_publish_order: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::local_workspace::{DevState, LocalWorkspaceOracle};

    println!("{}", "🏠 Local Workspace Oracle".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let mut oracle_ws = LocalWorkspaceOracle::new()?;
    oracle_ws.discover_projects()?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(oracle_ws.fetch_published_versions())?;

    let summary = oracle_ws.summary();
    let projects = oracle_ws.projects();

    local_print_summary(&summary);

    let filtered_projects: Vec<_> = if show_dirty {
        projects
            .values()
            .filter(|p| p.dev_state == DevState::Dirty)
            .collect()
    } else {
        projects.values().collect()
    };

    if show_dirty && !show_status {
        local_show_dirty_summary(&filtered_projects);
        return Ok(());
    }

    if show_status {
        match format {
            OracleOutputFormat::Code => {
                eprintln!("No code available for workspace status (try --format text)");
                std::process::exit(1);
            }
            OracleOutputFormat::Json => {
                let output = serde_json::json!({
                    "summary": summary,
                    "projects": projects,
                    "drift": oracle_ws.detect_drift(),
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
                println!("{}", "📦 Projects".bright_cyan().bold());
                println!("{}", "─".repeat(50).dimmed());

                let mut sorted_projects = filtered_projects.clone();
                sorted_projects.sort_by(|a, b| a.name.cmp(&b.name));

                for project in sorted_projects {
                    local_show_project_details(project);
                }
                println!();

                local_show_drift(&oracle_ws.detect_drift());
            }
        }
    }

    if show_publish_order {
        let order = oracle_ws.suggest_publish_order();

        match format {
            OracleOutputFormat::Code => {
                eprintln!("No code available for publish order (try --format text)");
                std::process::exit(1);
            }
            OracleOutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&order)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
                local_show_publish_order_text(&order);
            }
        }
    }

    if !show_status && !show_publish_order {
        local_show_usage();
    }

    Ok(())
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
            "✅ Index is current (no files changed since last index)"
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

/// Index stack documentation for RAG
pub fn cmd_oracle_rag_index(force: bool) -> anyhow::Result<()> {
    use oracle::rag::{
        fingerprint::DocumentFingerprint, persistence::RagPersistence, ChunkerConfig,
        HeijunkaReindexer, HybridRetriever, SemanticChunker,
    };

    println!("{}", "📚 RAG Indexer (Heijunka Mode)".bright_cyan().bold());
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
    ];

    let model_hash = [0u8; 32]; // BM25 has no model weights

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

/// Show RAG index statistics
pub fn cmd_oracle_rag_stats(format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::persistence::RagPersistence;

    println!("{}", "📊 RAG Index Statistics".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let persistence = RagPersistence::new();

    match persistence.stats()? {
        Some(manifest) => {
            match format {
                OracleOutputFormat::Code => {
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
                                "•".bright_blue(),
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

fn cookbook_show_recipe_not_found(id: &str, cookbook: &oracle::cookbook::Cookbook) {
    println!("{} Recipe '{}' not found", "Error:".bright_red().bold(), id);
    println!();
    println!("Available recipes:");
    for r in cookbook.recipes() {
        println!("  {} - {}", r.id.cyan(), r.title.dimmed());
    }
}

fn cookbook_show_help() {
    println!("{}", "📖 Batuta Cookbook".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();
    println!(
        "{}",
        "Practical recipes for common Sovereign AI Stack patterns".dimmed()
    );
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --cookbook".cyan(),
        "# List all recipes".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipe wasm-zero-js".cyan(),
        "# Show specific recipe".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipes-by-tag wasm".cyan(),
        "# Find by tag".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipes-by-component aprender".cyan(),
        "# Find by component".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --search-recipes \"random forest\"".cyan(),
        "# Search".dimmed()
    );
    println!();
    println!(
        "{} wasm, ml, distributed, quality, transpilation",
        "Tags:".bright_yellow()
    );
    println!();
}

fn cookbook_display_filter_results(
    recipes: &[&oracle::cookbook::Recipe],
    empty_msg: &str,
    title: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    if recipes.is_empty() {
        println!("{}", empty_msg);
    } else {
        display_recipe_list(recipes, title, format)?;
    }
    Ok(())
}

/// Handle cookbook commands
pub fn cmd_oracle_cookbook(
    list_all: bool,
    recipe_id: Option<String>,
    by_tag: Option<String>,
    by_component: Option<String>,
    search: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::cookbook::Cookbook;

    let cookbook = Cookbook::standard();

    if let Some(id) = recipe_id {
        return match cookbook.get(&id) {
            Some(recipe) => display_recipe(recipe, format),
            None => {
                cookbook_show_recipe_not_found(&id, &cookbook);
                Ok(())
            }
        };
    }

    if let Some(tag) = by_tag {
        let recipes = cookbook.find_by_tag(&tag);
        let empty_msg = format!(
            "No recipes found with tag '{}'\n\nAvailable tags: wasm, ml, distributed, quality, transpilation",
            tag
        );
        return cookbook_display_filter_results(
            &recipes,
            &empty_msg,
            &format!("Recipes tagged '{}'", tag),
            format,
        );
    }

    if let Some(component) = by_component {
        let recipes = cookbook.find_by_component(&component);
        return cookbook_display_filter_results(
            &recipes,
            &format!("No recipes found using component '{}'", component),
            &format!("Recipes using '{}'", component),
            format,
        );
    }

    if let Some(query) = search {
        let recipes = cookbook.search(&query);
        return cookbook_display_filter_results(
            &recipes,
            &format!("No recipes found matching '{}'", query),
            &format!("Recipes matching '{}'", query),
            format,
        );
    }

    if list_all {
        let recipes: Vec<_> = cookbook.recipes().iter().collect();
        display_recipe_list(&recipes, "All Cookbook Recipes", format)?;
        return Ok(());
    }

    cookbook_show_help();
    Ok(())
}

/// Format a slice of strings by wrapping each item with a prefix/suffix and joining with a separator.
fn format_items(items: &[String], prefix: &str, suffix: &str, sep: &str) -> String {
    items
        .iter()
        .map(|x| format!("{}{}{}", prefix, x, suffix))
        .collect::<Vec<_>>()
        .join(sep)
}

fn display_recipe_markdown(recipe: &oracle::cookbook::Recipe) {
    println!("# {}\n", recipe.title);
    println!("**ID:** `{}`\n", recipe.id);
    println!("## Problem\n\n{}\n", recipe.problem);
    println!(
        "## Components\n\n{}\n",
        format_items(&recipe.components, "`", "`", ", ")
    );
    println!(
        "## Tags\n\n{}\n",
        format_items(&recipe.tags, "`", "`", ", ")
    );
    println!("## Code\n\n```rust\n{}\n```\n", recipe.code);
    if !recipe.test_code.is_empty() {
        println!("## Tests\n\n```rust\n{}\n```\n", recipe.test_code);
    }
    if !recipe.related.is_empty() {
        println!(
            "## Related Recipes\n\n{}\n",
            format_items(&recipe.related, "`", "`", ", ")
        );
    }
}

fn display_recipe_code_line(line: &str) {
    if line.starts_with("//") || line.starts_with('#') {
        println!("{}", line.dimmed());
    } else if line.contains("fn ") || line.contains("pub ") || line.contains("use ") {
        println!("{}", line.bright_blue());
    } else {
        println!("{}", line);
    }
}

fn display_recipe_text(recipe: &oracle::cookbook::Recipe) {
    println!(
        "{} {}",
        "📖".bright_cyan(),
        recipe.title.bright_white().bold()
    );
    println!("{}", "─".repeat(60).dimmed());
    println!();
    println!("{} {}", "ID:".bright_yellow(), recipe.id.cyan());
    println!();
    println!("{}", "Problem:".bright_yellow());
    println!("  {}", recipe.problem);
    println!();
    println!("{}", "Components:".bright_yellow());
    for comp in &recipe.components {
        println!("  • {}", comp.cyan());
    }
    println!();
    println!("{}", "Tags:".bright_yellow());
    println!("  {}", format_items(&recipe.tags, "#", "", " ").dimmed());
    println!();
    println!("{}", "Code:".bright_yellow());
    println!("{}", "─".repeat(60).dimmed());
    for line in recipe.code.lines() {
        display_recipe_code_line(line);
    }
    println!("{}", "─".repeat(60).dimmed());
    if !recipe.test_code.is_empty() {
        println!();
        println!("{}", "Tests:".bright_yellow());
        println!("{}", "─".repeat(60).dimmed());
        for line in recipe.test_code.lines() {
            display_recipe_code_line(line);
        }
        println!("{}", "─".repeat(60).dimmed());
    }
    if !recipe.related.is_empty() {
        println!();
        println!("{}", "Related:".bright_yellow());
        for related in &recipe.related {
            println!("  → {}", related.cyan());
        }
    }
    println!();
}

/// Display a single recipe
fn display_recipe(
    recipe: &oracle::cookbook::Recipe,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Code => {
            println!("{}", recipe.code);
            if !recipe.test_code.is_empty() {
                println!("\n{}", recipe.test_code);
            }
        }
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(recipe)?);
        }
        OracleOutputFormat::Markdown => display_recipe_markdown(recipe),
        OracleOutputFormat::Text => display_recipe_text(recipe),
    }
    Ok(())
}

/// Display a list of recipes
fn display_recipe_list(
    recipes: &[&oracle::cookbook::Recipe],
    title: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Code => {
            for (i, recipe) in recipes.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                println!("// --- {} ---", recipe.id);
                println!("{}", recipe.code);
                if !recipe.test_code.is_empty() {
                    println!("\n{}", recipe.test_code);
                }
            }
        }
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(recipes)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("# {}\n", title);
            println!("| ID | Title | Components | Tags |");
            println!("|---|---|---|---|");
            for recipe in recipes {
                println!(
                    "| `{}` | {} | {} | {} |",
                    recipe.id,
                    recipe.title,
                    recipe.components.join(", "),
                    recipe.tags.join(", ")
                );
            }
        }
        OracleOutputFormat::Text => {
            println!("{} {}", "📖".bright_cyan(), title.bright_white().bold());
            println!("{}", "─".repeat(60).dimmed());
            println!();
            for recipe in recipes {
                println!(
                    "  {} {}",
                    recipe.id.cyan().bold(),
                    format!("- {}", recipe.title).dimmed()
                );
                println!(
                    "    {} {}",
                    "Components:".dimmed(),
                    recipe.components.join(", ").bright_blue()
                );
                println!(
                    "    {} {}",
                    "Tags:".dimmed(),
                    format_items(&recipe.tags, "#", "", " ")
                );
                println!();
            }
            println!(
                "{} Use {} to view a recipe",
                "Tip:".bright_yellow(),
                "--recipe <id>".cyan()
            );
            println!();
        }
    }
    Ok(())
}

// Classic oracle functions moved to oracle_classic.rs
pub use super::oracle_classic::{cmd_oracle, OracleOptions};
