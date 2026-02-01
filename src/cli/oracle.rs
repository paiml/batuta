//! Oracle command implementations
//!
//! This module contains all Oracle-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::oracle_indexing::{
    check_dir_for_changes, index_markdown_files, index_markdown_files_recursive,
    index_python_files, index_rust_files,
};

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

/// Index stack documentation for RAG
#[allow(clippy::cognitive_complexity)] // Sequential indexing steps are inherently complex
pub fn cmd_oracle_rag_index(force: bool) -> anyhow::Result<()> {
    use oracle::rag::{
        fingerprint::DocumentFingerprint,
        persistence::{CorpusSource, PersistedDocuments, RagPersistence},
        tui::inline,
        ChunkerConfig, HeijunkaReindexer, HybridRetriever, SemanticChunker,
    };
    use std::path::Path;

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
    ];

    // =========================================================================
    // Fingerprint-based change detection (Option 3)
    // Load existing fingerprints and check if any files changed.
    // If nothing changed, skip the entire reindex.
    // =========================================================================
    let model_hash = [0u8; 32]; // BM25 has no model weights

    if !force {
        if let Ok(Some((_, existing_docs, _))) = persistence.load() {
            if !existing_docs.fingerprints.is_empty() {
                println!(
                    "{}",
                    "Checking for changes against stored fingerprints...".dimmed()
                );

                let mut any_changed = false;
                let mut changed_count = 0usize;

                // Check all Rust stack dirs
                for dir in rust_stack_dirs.iter().chain(rust_corpus_dirs.iter()) {
                    if any_changed {
                        break;
                    }
                    let path = Path::new(dir);
                    if !path.exists() {
                        continue;
                    }
                    let component = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    // Check CLAUDE.md
                    let claude_md = path.join("CLAUDE.md");
                    if claude_md.exists() {
                        if let Ok(content) = std::fs::read_to_string(&claude_md) {
                            let doc_id = format!("{}/CLAUDE.md", component);
                            let current_fp = DocumentFingerprint::new(
                                content.as_bytes(),
                                &rust_chunker_config,
                                model_hash,
                            );
                            if let Some(stored_fp) = existing_docs.fingerprints.get(&doc_id) {
                                if stored_fp.needs_reindex(&current_fp) {
                                    any_changed = true;
                                    changed_count += 1;
                                }
                            } else {
                                any_changed = true; // New file
                                changed_count += 1;
                            }
                        }
                    }

                    // Check README.md
                    let readme_md = path.join("README.md");
                    if readme_md.exists() && !any_changed {
                        if let Ok(content) = std::fs::read_to_string(&readme_md) {
                            let doc_id = format!("{}/README.md", component);
                            let current_fp = DocumentFingerprint::new(
                                content.as_bytes(),
                                &rust_chunker_config,
                                model_hash,
                            );
                            if let Some(stored_fp) = existing_docs.fingerprints.get(&doc_id) {
                                if stored_fp.needs_reindex(&current_fp) {
                                    any_changed = true;
                                    changed_count += 1;
                                }
                            } else {
                                any_changed = true;
                                changed_count += 1;
                            }
                        }
                    }

                    // Check src/ directory for changes
                    if !any_changed {
                        let src_dir = path.join("src");
                        if src_dir.exists()
                            && check_dir_for_changes(
                                &src_dir,
                                src_dir.parent().unwrap_or(&src_dir),
                                component,
                                &rust_chunker_config,
                                model_hash,
                                &existing_docs.fingerprints,
                                "rs",
                            )
                        {
                            any_changed = true;
                            changed_count += 1;
                        }
                    }
                }

                // Check Python corpus dirs
                for dir in &python_corpus_dirs {
                    if any_changed {
                        break;
                    }
                    let path = Path::new(dir);
                    if !path.exists() {
                        continue;
                    }
                    let component = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    let claude_md = path.join("CLAUDE.md");
                    if claude_md.exists() {
                        if let Ok(content) = std::fs::read_to_string(&claude_md) {
                            let doc_id = format!("{}/CLAUDE.md", component);
                            let current_fp = DocumentFingerprint::new(
                                content.as_bytes(),
                                &python_chunker_config,
                                model_hash,
                            );
                            if let Some(stored_fp) = existing_docs.fingerprints.get(&doc_id) {
                                if stored_fp.needs_reindex(&current_fp) {
                                    any_changed = true;
                                    changed_count += 1;
                                }
                            } else {
                                any_changed = true;
                                changed_count += 1;
                            }
                        }
                    }

                    if !any_changed {
                        let src_dir = path.join("src");
                        if src_dir.exists()
                            && check_dir_for_changes(
                                &src_dir,
                                src_dir.parent().unwrap_or(&src_dir),
                                component,
                                &python_chunker_config,
                                model_hash,
                                &existing_docs.fingerprints,
                                "py",
                            )
                        {
                            any_changed = true;
                            changed_count += 1;
                        }
                    }
                }

                if !any_changed {
                    println!(
                        "{}",
                        "✅ Index is current (no files changed since last index)"
                            .bright_green()
                            .bold()
                    );
                    println!();
                    return Ok(());
                }

                println!(
                    "{} files changed, rebuilding index...",
                    changed_count.to_string().bright_yellow()
                );
                println!();
            }
        }
    }

    let rust_chunker = SemanticChunker::from_config(&rust_chunker_config);
    let python_chunker = SemanticChunker::from_config(&python_chunker_config);

    let mut reindexer = HeijunkaReindexer::new();
    let mut retriever = HybridRetriever::new();

    // Track sources for manifest
    let mut corpus_sources: Vec<CorpusSource> = Vec::new();

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
    for dir in &rust_stack_dirs {
        let path = Path::new(dir);
        if !path.exists() {
            continue;
        }

        let component = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Index CLAUDE.md (P0)
        let claude_md = path.join("CLAUDE.md");
        if claude_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&claude_md) {
                let doc_id = format!("{}/CLAUDE.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(content.as_bytes(), &rust_chunker_config, model_hash),
                );

                // Queue for reindexing (staleness = 0 for fresh)
                reindexer.enqueue(&doc_id, claude_md.clone(), 0);

                // Chunk and index
                let chunks = rust_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:20} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/CLAUDE.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index README.md (P1)
        let readme_md = path.join("README.md");
        if readme_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&readme_md) {
                let doc_id = format!("{}/README.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(content.as_bytes(), &rust_chunker_config, model_hash),
                );

                reindexer.enqueue(&doc_id, readme_md.clone(), 0);

                let chunks = rust_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:20} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/README.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index Rust source files (P2) - src/**/*.rs
        let src_dir = path.join("src");
        if src_dir.exists() {
            index_rust_files(
                &src_dir,
                src_dir.parent().unwrap_or(&src_dir),
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }

        // Index specification docs (P1)
        let specs_dir = path.join("docs/specifications");
        if specs_dir.exists() {
            index_markdown_files(
                &specs_dir,
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }

        // Index mdBook documentation (P1) - book/src/**/*.md
        let book_dir = path.join("book/src");
        if book_dir.exists() {
            index_markdown_files_recursive(
                &book_dir,
                book_dir.parent().unwrap_or(&book_dir),
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }
    }

    // Index Python ground truth corpora
    println!();
    println!("{}", "Scanning Python ground truth corpora...".dimmed());
    println!();

    for dir in &python_corpus_dirs {
        let path = Path::new(dir);
        if !path.exists() {
            println!("  {} {} (not found)", "⊘".dimmed(), dir.dimmed());
            continue;
        }

        let component = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Index CLAUDE.md (P0)
        let claude_md = path.join("CLAUDE.md");
        if claude_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&claude_md) {
                let doc_id = format!("{}/CLAUDE.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(
                        content.as_bytes(),
                        &python_chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, claude_md.clone(), 0);

                let chunks = python_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:30} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/CLAUDE.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index README.md (P1)
        let readme_md = path.join("README.md");
        if readme_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&readme_md) {
                let doc_id = format!("{}/README.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(
                        content.as_bytes(),
                        &python_chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, readme_md.clone(), 0);

                let chunks = python_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:30} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/README.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index Python source files (P2) - src/**/*.py
        let src_dir = path.join("src");
        if src_dir.exists() {
            index_python_files(
                &src_dir,
                src_dir.parent().unwrap_or(&src_dir),
                component,
                &python_chunker,
                &python_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }
    }

    // Index Rust ground truth corpora
    println!();
    println!("{}", "Scanning Rust ground truth corpora...".dimmed());
    println!();

    for dir in &rust_corpus_dirs {
        let path = Path::new(dir);
        if !path.exists() {
            println!("  {} {} (not found)", "⊘".dimmed(), dir.dimmed());
            continue;
        }

        let component = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Index CLAUDE.md (P0)
        let claude_md = path.join("CLAUDE.md");
        if claude_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&claude_md) {
                let doc_id = format!("{}/CLAUDE.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(content.as_bytes(), &rust_chunker_config, model_hash),
                );

                reindexer.enqueue(&doc_id, claude_md.clone(), 0);

                let chunks = rust_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:40} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/CLAUDE.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index README.md (P1)
        let readme_md = path.join("README.md");
        if readme_md.exists() {
            if let Ok(content) = std::fs::read_to_string(&readme_md) {
                let doc_id = format!("{}/README.md", component);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    DocumentFingerprint::new(content.as_bytes(), &rust_chunker_config, model_hash),
                );

                reindexer.enqueue(&doc_id, readme_md.clone(), 0);

                let chunks = rust_chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
                    retriever.index_document(&chunk_id, &chunk.content);
                    total_chunks += 1;
                }
                indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:40} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/README.md", component).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }

        // Index Rust source files (P2) - src/**/*.rs
        let src_dir = path.join("src");
        if src_dir.exists() {
            index_rust_files(
                &src_dir,
                src_dir.parent().unwrap_or(&src_dir),
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }

        // Index specification docs (P1)
        let specs_dir = path.join("docs/specifications");
        if specs_dir.exists() {
            index_markdown_files(
                &specs_dir,
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }

        // Index mdBook documentation (P1) - book/src/**/*.md
        let book_dir = path.join("book/src");
        if book_dir.exists() {
            index_markdown_files_recursive(
                &book_dir,
                book_dir.parent().unwrap_or(&book_dir),
                component,
                &rust_chunker,
                &rust_chunker_config,
                model_hash,
                &mut reindexer,
                &mut retriever,
                &mut indexed_count,
                &mut total_chunks,
                &mut fingerprints,
                &mut chunk_contents,
            );
        }
    }

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
    println!(
        "{}: {} unique terms",
        "Vocabulary".bright_yellow(),
        stats.total_terms
    );
    println!(
        "{}: {:.1} tokens",
        "Avg doc length".bright_yellow(),
        stats.avg_doc_length
    );
    println!();

    // Print reindexer stats
    let reindex_stats = reindexer.stats();
    println!(
        "{}: {} documents tracked",
        "Reindexer".bright_yellow(),
        reindex_stats.tracked_documents
    );
    println!();

    // Build corpus sources for manifest
    // Group documents by component
    let mut component_stats: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();

    // Count chunks per component from indexed data
    for dir in rust_stack_dirs.iter().chain(python_corpus_dirs.iter()) {
        let path = Path::new(dir);
        if path.exists() {
            let component = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            // Increment by 1 for each indexed document
            component_stats.entry(component).or_insert((0, 0));
        }
    }

    // Create corpus sources (simplified - count from stats)
    corpus_sources.push(CorpusSource {
        id: "sovereign-ai-stack".to_string(),
        commit: None, // Could extract from git if desired
        doc_count: indexed_count,
        chunk_count: total_chunks,
    });

    // Save to disk
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
                    println!(
                        "{}: {}",
                        "Index version".bright_yellow(),
                        manifest.version.cyan()
                    );
                    println!(
                        "{}: {}",
                        "Batuta version".bright_yellow(),
                        manifest.batuta_version.cyan()
                    );
                    println!(
                        "{}: {}",
                        "Indexed".bright_yellow(),
                        format_timestamp(manifest.indexed_at).cyan()
                    );
                    println!(
                        "{}: {:?}",
                        "Cache path".bright_yellow(),
                        persistence.cache_path()
                    );
                    println!();

                    // Calculate totals
                    let total_docs: usize = manifest.sources.iter().map(|s| s.doc_count).sum();
                    let total_chunks: usize = manifest.sources.iter().map(|s| s.chunk_count).sum();

                    println!("{}: {} documents", "Total".bright_yellow(), total_docs);
                    println!("{}: {} chunks", "Total".bright_yellow(), total_chunks);
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

    // Show specific recipe
    if let Some(id) = recipe_id {
        if let Some(recipe) = cookbook.get(&id) {
            display_recipe(recipe, format)?;
        } else {
            println!("{} Recipe '{}' not found", "Error:".bright_red().bold(), id);
            println!();
            println!("Available recipes:");
            for r in cookbook.recipes() {
                println!("  {} - {}", r.id.cyan(), r.title.dimmed());
            }
        }
        return Ok(());
    }

    // Search by tag
    if let Some(tag) = by_tag {
        let recipes = cookbook.find_by_tag(&tag);
        if recipes.is_empty() {
            println!("No recipes found with tag '{}'", tag);
            println!();
            println!("Available tags: wasm, ml, distributed, quality, transpilation");
        } else {
            display_recipe_list(&recipes, &format!("Recipes tagged '{}'", tag), format)?;
        }
        return Ok(());
    }

    // Search by component
    if let Some(component) = by_component {
        let recipes = cookbook.find_by_component(&component);
        if recipes.is_empty() {
            println!("No recipes found using component '{}'", component);
        } else {
            display_recipe_list(&recipes, &format!("Recipes using '{}'", component), format)?;
        }
        return Ok(());
    }

    // Search by keyword
    if let Some(query) = search {
        let recipes = cookbook.search(&query);
        if recipes.is_empty() {
            println!("No recipes found matching '{}'", query);
        } else {
            display_recipe_list(&recipes, &format!("Recipes matching '{}'", query), format)?;
        }
        return Ok(());
    }

    // List all recipes
    if list_all {
        let recipes: Vec<_> = cookbook.recipes().iter().collect();
        display_recipe_list(&recipes, "All Cookbook Recipes", format)?;
        return Ok(());
    }

    // Default: show cookbook help
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

    Ok(())
}

/// Display a single recipe
fn display_recipe(
    recipe: &oracle::cookbook::Recipe,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(recipe)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("# {}\n", recipe.title);
            println!("**ID:** `{}`\n", recipe.id);
            println!("## Problem\n\n{}\n", recipe.problem);
            println!(
                "## Components\n\n{}\n",
                recipe
                    .components
                    .iter()
                    .map(|c| format!("`{}`", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "## Tags\n\n{}\n",
                recipe
                    .tags
                    .iter()
                    .map(|t| format!("`{}`", t))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("## Code\n\n```rust\n{}\n```\n", recipe.code);
            if !recipe.related.is_empty() {
                println!(
                    "## Related Recipes\n\n{}\n",
                    recipe
                        .related
                        .iter()
                        .map(|r| format!("`{}`", r))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        OracleOutputFormat::Text => {
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
            println!(
                "  {}",
                recipe
                    .tags
                    .iter()
                    .map(|t| format!("#{}", t))
                    .collect::<Vec<_>>()
                    .join(" ")
                    .dimmed()
            );
            println!();
            println!("{}", "Code:".bright_yellow());
            println!("{}", "─".repeat(60).dimmed());
            // Syntax highlight hint for code blocks
            for line in recipe.code.lines() {
                if line.starts_with("//") || line.starts_with('#') {
                    println!("{}", line.dimmed());
                } else if line.contains("fn ") || line.contains("pub ") || line.contains("use ") {
                    println!("{}", line.bright_blue());
                } else {
                    println!("{}", line);
                }
            }
            println!("{}", "─".repeat(60).dimmed());
            if !recipe.related.is_empty() {
                println!();
                println!("{}", "Related:".bright_yellow());
                for related in &recipe.related {
                    println!("  → {}", related.cyan());
                }
            }
            println!();
        }
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
                    recipe
                        .tags
                        .iter()
                        .map(|t| format!("#{}", t))
                        .collect::<Vec<_>>()
                        .join(" ")
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
