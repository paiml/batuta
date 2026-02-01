//! Oracle command implementations
//!
//! This module contains all Oracle-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

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

/// RAG-based query using indexed documentation
pub fn cmd_oracle_rag(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::{persistence::RagPersistence, tui::inline, DocumentIndex, HybridRetriever};

    println!("{}", "🔍 RAG Oracle Mode".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Try to load persisted index
    let persistence = RagPersistence::new();
    let (retriever, doc_count, chunk_count) = match persistence.load() {
        Ok(Some((persisted_index, persisted_docs, manifest))) => {
            let retriever = HybridRetriever::from_persisted(persisted_index);
            let doc_count = persisted_docs.documents.len();
            let chunk_count = persisted_docs.total_chunks;

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

            (retriever, doc_count, chunk_count)
        }
        Ok(None) => {
            println!(
                "{}",
                "No cached index found. Run 'batuta oracle --rag-index' first."
                    .bright_yellow()
                    .bold()
            );
            println!();
            return Ok(());
        }
        Err(e) => {
            println!(
                "{}: {} - rebuilding index recommended",
                "Warning".bright_yellow(),
                e
            );
            println!();
            return Ok(());
        }
    };

    // Show index status
    println!(
        "{}: {} documents, {} chunks",
        "Index".bright_yellow(),
        doc_count,
        chunk_count
    );
    println!();

    if let Some(query_text) = query {
        let empty_index = DocumentIndex::default();
        let results = retriever.retrieve(&query_text, &empty_index, 10);

        if results.is_empty() {
            println!(
                "{}",
                "No results found. Try running --rag-index first.".dimmed()
            );
            return Ok(());
        }

        match format {
            OracleOutputFormat::Json => {
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
            }
            OracleOutputFormat::Markdown => {
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
            OracleOutputFormat::Text => {
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
                        // Show first 200 chars of content
                        let preview: String = result.content.chars().take(200).collect();
                        println!("   {}", preview.dimmed());
                    }
                    println!();
                }
            }
        }
    } else {
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

    Ok(())
}

// ============================================================================
// Local Workspace Oracle Commands
// ============================================================================

/// Local workspace discovery and multi-project intelligence
pub fn cmd_oracle_local(
    show_status: bool,
    show_dirty: bool,
    show_publish_order: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::local_workspace::{DevState, DriftType, LocalWorkspaceOracle};

    println!("{}", "🏠 Local Workspace Oracle".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let mut oracle = LocalWorkspaceOracle::new()?;
    oracle.discover_projects()?;

    // Fetch published versions (blocking for now)
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(oracle.fetch_published_versions())?;

    let summary = oracle.summary();
    let projects = oracle.projects();

    // Print summary
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

    // Filter projects if --dirty flag
    let filtered_projects: Vec<_> = if show_dirty {
        projects
            .values()
            .filter(|p| p.dev_state == DevState::Dirty)
            .collect()
    } else {
        projects.values().collect()
    };

    if show_dirty && !show_status {
        // Just show dirty projects summary
        println!(
            "{} {} projects with uncommitted changes:",
            "🔴".bright_red(),
            filtered_projects.len()
        );
        println!();
        for project in &filtered_projects {
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
        return Ok(());
    }

    if show_status {
        match format {
            OracleOutputFormat::Json => {
                let output = serde_json::json!({
                    "summary": summary,
                    "projects": projects,
                    "drift": oracle.detect_drift(),
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
                // Show project details
                println!("{}", "📦 Projects".bright_cyan().bold());
                println!("{}", "─".repeat(50).dimmed());

                let mut sorted_projects = filtered_projects.clone();
                sorted_projects.sort_by(|a, b| a.name.cmp(&b.name));

                for project in sorted_projects {
                    let (status_icon, state_label) = match project.dev_state {
                        DevState::Dirty => ("●".bright_red(), "DIRTY".bright_red()),
                        DevState::Unpushed => ("◐".bright_yellow(), "UNPUSHED".bright_yellow()),
                        DevState::Clean => ("○".bright_green(), "clean".bright_green()),
                    };

                    let version_info = match &project.published_version {
                        Some(pub_v) if pub_v == &project.local_version => {
                            format!("v{}", project.local_version)
                                .bright_green()
                                .to_string()
                        }
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
                println!();

                // Show version drift
                let drifts = oracle.detect_drift();
                if !drifts.is_empty() {
                    println!("{}", "📊 Version Drift".bright_cyan().bold());
                    println!("{}", "─".repeat(50).dimmed());

                    for drift in &drifts {
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
            }
        }
    }

    if show_publish_order {
        let order = oracle.suggest_publish_order();

        match format {
            OracleOutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&order)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
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

                let needs_publish: Vec<_> =
                    order.order.iter().filter(|s| s.needs_publish).collect();

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
        }
    }

    // Show usage hints if neither flag specified
    if !show_status && !show_publish_order {
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
        documents: std::collections::HashMap::new(), // Document content not persisted in this version
        fingerprints,
        total_chunks,
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

/// Check if any files in a directory have changed compared to stored fingerprints.
/// Returns true on first changed or new file found (fast short-circuit).
fn check_dir_for_changes(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    use oracle::rag::fingerprint::DocumentFingerprint;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with('.')
                    && name != "target"
                    && name != "__pycache__"
                    && check_dir_for_changes(
                        &path,
                        base_dir,
                        component,
                        chunker_config,
                        model_hash,
                        existing_fingerprints,
                        extension,
                    )
                {
                    return true;
                }
            }
        } else if path.extension().is_some_and(|ext| ext == extension) {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if content.trim().is_empty() || content.lines().count() < 5 {
                    continue;
                }

                let relative_path = path.strip_prefix(base_dir).unwrap_or(&path);
                let doc_id = format!("{}/{}", component, relative_path.display());

                let current_fp =
                    DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash);

                match existing_fingerprints.get(&doc_id) {
                    Some(stored_fp) if !stored_fp.needs_reindex(&current_fp) => {
                        // Unchanged, keep checking
                    }
                    _ => return true, // Changed or new file
                }
            }
        }
    }

    false
}

/// Recursively index Python files in a directory
#[allow(clippy::too_many_arguments)]
fn index_python_files(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) {
    use oracle::rag::tui::inline;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Skip __pycache__ and hidden directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with('.') && name != "__pycache__" {
                    index_python_files(
                        &path,
                        base_dir,
                        component,
                        chunker,
                        chunker_config,
                        model_hash,
                        reindexer,
                        retriever,
                        indexed_count,
                        total_chunks,
                        fingerprints,
                    );
                }
            }
        } else if path.extension().is_some_and(|ext| ext == "py") {
            // Index .py files
            if let Ok(content) = std::fs::read_to_string(&path) {
                // Skip empty or trivial files
                if content.trim().is_empty() || content.lines().count() < 5 {
                    continue;
                }

                let relative_path = path.strip_prefix(base_dir).unwrap_or(&path);
                let doc_id = format!("{}/{}", component, relative_path.display());

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    oracle::rag::DocumentFingerprint::new(
                        content.as_bytes(),
                        chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, path.clone(), 0);

                let chunks = chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    retriever.index_document(&chunk_id, &chunk.content);
                    *total_chunks += 1;
                }
                *indexed_count += 1;

                // Only print for files with substantial content
                if chunks.len() > 1 {
                    let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                    println!(
                        "  {} {:40} {} ({} chunks)",
                        "✓".bright_green(),
                        relative_path.display().to_string().cyan(),
                        bar,
                        chunks.len()
                    );
                }
            }
        }
    }
}

/// Recursively index Rust source files in a directory
#[allow(clippy::too_many_arguments)]
fn index_rust_files(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) {
    use oracle::rag::tui::inline;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Skip target and hidden directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with('.') && name != "target" {
                    index_rust_files(
                        &path,
                        base_dir,
                        component,
                        chunker,
                        chunker_config,
                        model_hash,
                        reindexer,
                        retriever,
                        indexed_count,
                        total_chunks,
                        fingerprints,
                    );
                }
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            // Index .rs files
            if let Ok(content) = std::fs::read_to_string(&path) {
                // Skip empty or trivial files
                if content.trim().is_empty() || content.lines().count() < 5 {
                    continue;
                }

                let relative_path = path.strip_prefix(base_dir).unwrap_or(&path);
                let doc_id = format!("{}/{}", component, relative_path.display());

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    oracle::rag::DocumentFingerprint::new(
                        content.as_bytes(),
                        chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, path.clone(), 0);

                let chunks = chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    retriever.index_document(&chunk_id, &chunk.content);
                    *total_chunks += 1;
                }
                *indexed_count += 1;

                // Only print for files with substantial content
                if chunks.len() > 1 {
                    let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                    println!(
                        "  {} {:40} {} ({} chunks)",
                        "✓".bright_green(),
                        relative_path.display().to_string().cyan(),
                        bar,
                        chunks.len()
                    );
                }
            }
        }
    }
}

/// Index markdown files in a directory (non-recursive)
#[allow(clippy::too_many_arguments)]
fn index_markdown_files(
    dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) {
    use oracle::rag::tui::inline;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "md") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                // Skip empty files
                if content.trim().is_empty() {
                    continue;
                }

                let file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file.md");
                let doc_id = format!("{}/docs/{}", component, file_name);

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    oracle::rag::DocumentFingerprint::new(
                        content.as_bytes(),
                        chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, path.clone(), 0);

                let chunks = chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    retriever.index_document(&chunk_id, &chunk.content);
                    *total_chunks += 1;
                }
                *indexed_count += 1;

                let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                println!(
                    "  {} {:40} {} ({} chunks)",
                    "✓".bright_green(),
                    format!("{}/docs/{}", component, file_name).cyan(),
                    bar,
                    chunks.len()
                );
            }
        }
    }
}

/// Recursively index markdown files in a directory (for mdBook)
#[allow(clippy::too_many_arguments)]
fn index_markdown_files_recursive(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) {
    use oracle::rag::tui::inline;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with('.') {
                    index_markdown_files_recursive(
                        &path,
                        base_dir,
                        component,
                        chunker,
                        chunker_config,
                        model_hash,
                        reindexer,
                        retriever,
                        indexed_count,
                        total_chunks,
                        fingerprints,
                    );
                }
            }
        } else if path.extension().is_some_and(|ext| ext == "md") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                // Skip empty or SUMMARY files
                if content.trim().is_empty() {
                    continue;
                }

                let relative_path = path.strip_prefix(base_dir).unwrap_or(&path);
                let doc_id = format!("{}/book/{}", component, relative_path.display());

                // Record fingerprint
                fingerprints.insert(
                    doc_id.clone(),
                    oracle::rag::DocumentFingerprint::new(
                        content.as_bytes(),
                        chunker_config,
                        model_hash,
                    ),
                );

                reindexer.enqueue(&doc_id, path.clone(), 0);

                let chunks = chunker.split(&content);
                for chunk in &chunks {
                    let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
                    retriever.index_document(&chunk_id, &chunk.content);
                    *total_chunks += 1;
                }
                *indexed_count += 1;

                // Only print for files with substantial content
                if chunks.len() > 1 {
                    let bar = inline::bar(chunks.len() as f64, 20.0, 15);
                    println!(
                        "  {} {:40} {} ({} chunks)",
                        "✓".bright_green(),
                        relative_path.display().to_string().cyan(),
                        bar,
                        chunks.len()
                    );
                }
            }
        }
    }
}

/// Show RAG dashboard (TUI)
#[cfg(feature = "native")]
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

#[allow(clippy::too_many_arguments)]
pub fn cmd_oracle(
    query: Option<String>,
    recommend: bool,
    problem: Option<String>,
    data_size: Option<String>,
    integrate: Option<String>,
    capabilities: Option<String>,
    list: bool,
    show: Option<String>,
    interactive: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::{OracleQuery, Recommender};

    let recommender = Recommender::new();

    // List all components
    if list {
        display_component_list(&recommender, format)?;
        return Ok(());
    }

    // Show component details
    if let Some(component_name) = show {
        display_component_details(&recommender, &component_name, format)?;
        return Ok(());
    }

    // Show capabilities
    if let Some(component_name) = capabilities {
        display_capabilities(&recommender, &component_name, format)?;
        return Ok(());
    }

    // Show integration pattern
    if let Some(components) = integrate {
        display_integration(&recommender, &components, format)?;
        return Ok(());
    }

    // Interactive mode
    if interactive {
        run_interactive_oracle(&recommender)?;
        return Ok(());
    }

    // Query mode
    if let Some(query_text) = query {
        // Parse data size if provided
        let parsed_size = data_size.and_then(|s| parse_data_size(&s));

        // Build query
        let mut oracle_query = OracleQuery::new(&query_text);
        if let Some(size) = parsed_size {
            oracle_query = oracle_query.with_data_size(size);
        }

        // Get recommendation
        let response = recommender.query_structured(&oracle_query);
        display_oracle_response(&response, format)?;
        return Ok(());
    }

    // Recommendation mode
    if recommend {
        let query_text = problem.unwrap_or_else(|| "general ML task".into());
        let response = recommender.query(&query_text);
        display_oracle_response(&response, format)?;
        return Ok(());
    }

    // Default: show help
    println!("{}", "🔮 Batuta Oracle Mode".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();
    println!(
        "{}",
        "Query the Sovereign AI Stack for recommendations".dimmed()
    );
    println!();
    println!("{}", "Knowledge Graph:".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle".cyan(),
        "\"How do I train a random forest?\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recommend --problem".cyan(),
        "\"image classification\"".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --capabilities".cyan(),
        "aprender".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --integrate".cyan(),
        "\"aprender,realizar\"".dimmed()
    );
    println!("  {} {}", "batuta oracle --list".cyan(), "".dimmed());
    println!("  {} {}", "batuta oracle --interactive".cyan(), "".dimmed());
    println!();
    println!("{}", "Cookbook (Practical Recipes):".bright_yellow());
    println!(
        "  {} {}",
        "batuta oracle --cookbook".cyan(),
        "# List all recipes".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipe".cyan(),
        "wasm-zero-js       # Show recipe".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --recipes-by-tag".cyan(),
        "ml        # By tag".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --search-recipes".cyan(),
        "\"gpu\"   # Search".dimmed()
    );
    println!();

    Ok(())
}

fn display_component_list(
    recommender: &oracle::Recommender,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "🔮 Sovereign AI Stack Components".bright_cyan().bold()
    );
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let components: Vec<_> = recommender.list_components();

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&components)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Sovereign AI Stack Components\n");
            println!("| Component | Version | Layer | Description |");
            println!("|-----------|---------|-------|-------------|");
            for name in &components {
                if let Some(comp) = recommender.get_component(name) {
                    println!(
                        "| {} | {} | {} | {} |",
                        comp.name, comp.version, comp.layer, comp.description
                    );
                }
            }
        }
        OracleOutputFormat::Text => {
            // Group by layer
            for layer in oracle::StackLayer::all() {
                let layer_components: Vec<_> = components
                    .iter()
                    .filter_map(|name| recommender.get_component(name))
                    .filter(|c| c.layer == layer)
                    .collect();

                if !layer_components.is_empty() {
                    println!("{} {}", "Layer".bold(), format!("{}", layer).cyan());
                    for comp in layer_components {
                        println!(
                            "  {} {} {} - {}",
                            "•".bright_blue(),
                            comp.name.bright_green(),
                            format!("v{}", comp.version).dimmed(),
                            comp.description.dimmed()
                        );
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn display_component_details(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let Some(comp) = recommender.get_component(name) else {
        println!("{} Component '{}' not found", "❌".red(), name);
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&comp)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## {}\n", comp.name);
            println!("**Version:** {}\n", comp.version);
            println!("**Layer:** {}\n", comp.layer);
            println!("**Description:** {}\n", comp.description);
            println!("### Capabilities\n");
            for cap in &comp.capabilities {
                println!(
                    "- **{}**{}",
                    cap.name,
                    cap.description
                        .as_ref()
                        .map(|d| format!(": {}", d))
                        .unwrap_or_default()
                );
            }
        }
        OracleOutputFormat::Text => {
            println!("{}", format!("📦 {}", comp.name).bright_cyan().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Version".bold(), comp.version.cyan());
            println!("{}: {}", "Layer".bold(), format!("{}", comp.layer).cyan());
            println!("{}: {}", "Description".bold(), comp.description);
            println!();
            println!("{}", "Capabilities:".bright_yellow());
            for cap in &comp.capabilities {
                let desc = cap
                    .description
                    .as_ref()
                    .map(|d| format!(" - {}", d.dimmed()))
                    .unwrap_or_default();
                println!("  {} {}{}", "•".bright_blue(), cap.name.green(), desc);
            }
            println!();
        }
    }

    Ok(())
}

fn display_capabilities(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let caps = recommender.get_capabilities(name);

    if caps.is_empty() {
        println!("{} No capabilities found for '{}'", "❌".red(), name);
        return Ok(());
    }

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&caps)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Capabilities of {}\n", name);
            for cap in &caps {
                println!("- {}", cap);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔧 Capabilities of {}", name).bright_cyan().bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            for cap in &caps {
                println!("  {} {}", "•".bright_blue(), cap.green());
            }
            println!();
        }
    }

    Ok(())
}

fn display_integration(
    recommender: &oracle::Recommender,
    components: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        println!(
            "{} Please specify two components separated by comma",
            "❌".red()
        );
        println!(
            "  Example: {} {}",
            "batuta oracle --integrate".cyan(),
            "\"aprender,realizar\"".dimmed()
        );
        return Ok(());
    }

    let from = parts[0];
    let to = parts[1];

    let Some(pattern) = recommender.get_integration(from, to) else {
        println!(
            "{} No integration pattern found from '{}' to '{}'",
            "❌".red(),
            from,
            to
        );
        return Ok(());
    };

    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&pattern)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Integration: {} → {}\n", from, to);
            println!("**Pattern:** {}\n", pattern.pattern_name);
            println!("**Description:** {}\n", pattern.description);
            if let Some(template) = &pattern.code_template {
                println!("### Code Example\n");
                println!("```rust\n{}\n```", template);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("🔗 Integration: {} → {}", from, to)
                    .bright_cyan()
                    .bold()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!();
            println!("{}: {}", "Pattern".bold(), pattern.pattern_name.cyan());
            println!("{}: {}", "Description".bold(), pattern.description);
            println!();
            if let Some(template) = &pattern.code_template {
                println!("{}", "Code Example:".bright_yellow());
                println!("{}", "─".repeat(40).dimmed());
                for line in template.lines() {
                    println!("  {}", line.dimmed());
                }
                println!("{}", "─".repeat(40).dimmed());
            }
            println!();
        }
    }

    Ok(())
}

pub fn display_oracle_response(
    response: &oracle::OracleResponse,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&response)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => {
            println!("## Oracle Recommendation\n");
            println!("**Problem Class:** {}\n", response.problem_class);
            if let Some(algo) = &response.algorithm {
                println!("**Algorithm:** {}\n", algo);
            }
            println!("### Primary Recommendation\n");
            println!("- **Component:** {}", response.primary.component);
            if let Some(path) = &response.primary.path {
                println!("- **Module:** `{}`", path);
            }
            println!(
                "- **Confidence:** {:.0}%",
                response.primary.confidence * 100.0
            );
            println!("- **Rationale:** {}\n", response.primary.rationale);

            if !response.supporting.is_empty() {
                println!("### Supporting Components\n");
                for rec in &response.supporting {
                    println!(
                        "- **{}** ({:.0}%): {}",
                        rec.component,
                        rec.confidence * 100.0,
                        rec.rationale
                    );
                }
                println!();
            }

            println!("### Compute Backend\n");
            println!("- **Backend:** {}", response.compute.backend);
            println!("- **Rationale:** {}\n", response.compute.rationale);

            if response.distribution.needed {
                println!("### Distribution\n");
                println!(
                    "- **Tool:** {}",
                    response.distribution.tool.as_deref().unwrap_or("N/A")
                );
                println!("- **Rationale:** {}\n", response.distribution.rationale);
            }

            if let Some(code) = &response.code_example {
                println!("### Code Example\n");
                println!("```rust\n{}\n```\n", code);
            }
        }
        OracleOutputFormat::Text => {
            println!();
            println!("{}", "🔮 Oracle Recommendation".bright_cyan().bold());
            println!("{}", "═".repeat(60).dimmed());
            println!();

            // Problem classification
            println!(
                "{} {}: {}",
                "📊".bright_blue(),
                "Problem Class".bold(),
                response.problem_class.cyan()
            );
            if let Some(algo) = &response.algorithm {
                println!(
                    "{} {}: {}",
                    "🧮".bright_blue(),
                    "Algorithm".bold(),
                    algo.cyan()
                );
            }
            println!();

            // Primary recommendation
            println!("{}", "🎯 Primary Recommendation".bright_yellow().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!(
                "  {}: {}",
                "Component".bold(),
                response.primary.component.bright_green()
            );
            if let Some(path) = &response.primary.path {
                println!("  {}: {}", "Module".bold(), path.cyan());
            }
            println!(
                "  {}: {}",
                "Confidence".bold(),
                format!("{:.0}%", response.primary.confidence * 100.0).bright_green()
            );
            println!(
                "  {}: {}",
                "Rationale".bold(),
                response.primary.rationale.dimmed()
            );
            println!();

            // Supporting components
            if !response.supporting.is_empty() {
                println!("{}", "🔧 Supporting Components".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                for rec in &response.supporting {
                    println!(
                        "  {} {} ({:.0}%)",
                        "•".bright_blue(),
                        rec.component.green(),
                        rec.confidence * 100.0
                    );
                    println!("    {}", rec.rationale.dimmed());
                }
                println!();
            }

            // Compute backend
            println!("{}", "⚡ Compute Backend".bright_yellow().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!(
                "  {}: {}",
                "Backend".bold(),
                format!("{}", response.compute.backend).bright_green()
            );
            println!("  {}", response.compute.rationale.dimmed());
            println!();

            // Distribution
            if response.distribution.needed {
                println!("{}", "🌐 Distribution".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                println!(
                    "  {}: {}",
                    "Tool".bold(),
                    response
                        .distribution
                        .tool
                        .as_deref()
                        .unwrap_or("N/A")
                        .bright_green()
                );
                if let Some(nodes) = response.distribution.node_count {
                    println!("  {}: {}", "Nodes".bold(), nodes);
                }
                println!("  {}", response.distribution.rationale.dimmed());
                println!();
            }

            // Code example
            if let Some(code) = &response.code_example {
                println!("{}", "💡 Example Code".bright_yellow().bold());
                println!("{}", "─".repeat(50).dimmed());
                for line in code.lines() {
                    println!("  {}", line.dimmed());
                }
                println!("{}", "─".repeat(50).dimmed());
                println!();
            }

            // Related queries
            if !response.related_queries.is_empty() {
                println!("{}", "❓ Related Queries".bright_yellow());
                for query in &response.related_queries {
                    println!("  {} {}", "→".bright_blue(), query.dimmed());
                }
                println!();
            }

            println!("{}", "═".repeat(60).dimmed());
        }
    }

    Ok(())
}

fn run_interactive_oracle(recommender: &oracle::Recommender) -> anyhow::Result<()> {
    use std::io::{self, Write};

    println!();
    println!("{}", "🔮 Batuta Oracle Mode v1.0".bright_cyan().bold());
    println!(
        "{}",
        "   Ask questions about the Sovereign AI Stack".dimmed()
    );
    println!("{}", "   Type 'exit' or 'quit' to leave".dimmed());
    println!();

    loop {
        print!("{} ", ">".bright_cyan());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            println!();
            println!("{}", "👋 Goodbye!".bright_cyan());
            break;
        }

        if input == "help" {
            println!();
            println!("{}", "Commands:".bright_yellow());
            println!("  {} - Ask a question about the stack", "any text".cyan());
            println!("  {} - List all components", "list".cyan());
            println!("  {} - Show component details", "show <component>".cyan());
            println!("  {} - Show capabilities", "caps <component>".cyan());
            println!("  {} - Exit interactive mode", "exit".cyan());
            println!();
            continue;
        }

        if input == "list" {
            display_component_list(recommender, OracleOutputFormat::Text)?;
            continue;
        }

        if input.starts_with("show ") {
            let name = input
                .strip_prefix("show ")
                .expect("prefix verified by starts_with")
                .trim();
            display_component_details(recommender, name, OracleOutputFormat::Text)?;
            continue;
        }

        if input.starts_with("caps ") {
            let name = input
                .strip_prefix("caps ")
                .expect("prefix verified by starts_with")
                .trim();
            display_capabilities(recommender, name, OracleOutputFormat::Text)?;
            continue;
        }

        // Process as query
        let response = recommender.query(input);
        display_oracle_response(&response, OracleOutputFormat::Text)?;
    }

    Ok(())
}

fn parse_data_size(s: &str) -> Option<oracle::DataSize> {
    super::parse_data_size_value(s).map(oracle::DataSize::samples)
}
