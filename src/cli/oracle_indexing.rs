//! RAG Indexing Functions
//!
//! This module contains file indexing helpers for the RAG system.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

/// Check if a directory name should be skipped during indexing (all languages)
fn should_skip_directory(name: &str) -> bool {
    name.starts_with('.') || name == "target" || name == "__pycache__"
}

/// Check if a directory should be skipped for Python indexing
fn should_skip_python_dir(name: &str) -> bool {
    name.starts_with('.') || name == "__pycache__"
}

/// Check if a directory should be skipped for Rust indexing
fn should_skip_rust_dir(name: &str) -> bool {
    name.starts_with('.') || name == "target"
}

/// Check if a directory should be skipped for markdown (hidden only)
fn should_skip_hidden_dir(name: &str) -> bool {
    name.starts_with('.')
}

/// Check if a directory should be recursed into for indexing
fn should_recurse_dir(path: &std::path::Path, skip_fn: fn(&str) -> bool) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| !skip_fn(name))
}

/// Index chunks from a file and update retriever/counters, returns chunk count
#[allow(clippy::too_many_arguments)]
fn index_file_chunks(
    content: &str,
    doc_id: &str,
    chunker: &oracle::rag::SemanticChunker,
    retriever: &mut oracle::rag::HybridRetriever,
    total_chunks: &mut usize,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) -> usize {
    let chunks = chunker.split(content);
    let chunk_count = chunks.len();
    for chunk in &chunks {
        let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
        chunk_contents.insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
        retriever.index_document(&chunk_id, &chunk.content);
        *total_chunks += 1;
    }
    chunk_count
}

/// Print progress for indexed file (only if > 1 chunk)
fn print_file_indexed(relative_path: &std::path::Path, chunks_len: usize) {
    use oracle::rag::tui::inline;

    if chunks_len > 1 {
        let bar = inline::bar(chunks_len as f64, 20.0, 15);
        println!(
            "  {} {:40} {} ({} chunks)",
            "✓".bright_green(),
            relative_path.display().to_string().cyan(),
            bar,
            chunks_len
        );
    }
}

/// Print progress for indexed markdown doc (always shows)
fn print_markdown_indexed(display_path: &str, chunks_len: usize) {
    use oracle::rag::tui::inline;

    let bar = inline::bar(chunks_len as f64, 20.0, 15);
    println!(
        "  {} {:40} {} ({} chunks)",
        "✓".bright_green(),
        display_path.cyan(),
        bar,
        chunks_len
    );
}

/// Check if file content is trivial (empty or too short)
fn is_trivial_content(content: &str) -> bool {
    content.trim().is_empty() || content.lines().count() < 5
}

/// Index a single named document (e.g., CLAUDE.md, README.md) with fingerprint + print.
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_doc_file(
    file_path: &std::path::Path,
    doc_id: &str,
    display_name: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return;
    };

    fingerprints.insert(
        doc_id.to_string(),
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
    );
    reindexer.enqueue(doc_id, file_path.to_path_buf(), 0);

    let chunk_count = index_file_chunks(
        &content, doc_id, chunker, retriever, total_chunks, chunk_contents,
    );
    *indexed_count += 1;

    print_markdown_indexed(display_name, chunk_count);
}

/// Check if a single document's fingerprint has changed compared to stored fingerprints.
pub(crate) fn doc_fingerprint_changed(
    file_path: &std::path::Path,
    doc_id: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> bool {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return false;
    };
    let current_fp =
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash);
    match existing_fingerprints.get(doc_id) {
        Some(stored_fp) if !stored_fp.needs_reindex(&current_fp) => false,
        Some(_) => true,
        None => true, // New file
    }
}

/// Index all files for a single component directory (CLAUDE.md, README.md, src/, specs/, book/).
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_component(
    path: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    extension: &str,
    include_specs: bool,
    include_book: bool,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    // Index CLAUDE.md (P0)
    let claude_md = path.join("CLAUDE.md");
    if claude_md.exists() {
        let doc_id = format!("{}/CLAUDE.md", component);
        index_doc_file(
            &claude_md, &doc_id, &doc_id, chunker, chunker_config, model_hash,
            reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
        );
    }

    // Index README.md (P1)
    let readme_md = path.join("README.md");
    if readme_md.exists() {
        let doc_id = format!("{}/README.md", component);
        index_doc_file(
            &readme_md, &doc_id, &doc_id, chunker, chunker_config, model_hash,
            reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
        );
    }

    // Index source files (P2)
    let src_dir = path.join("src");
    if src_dir.exists() {
        let base = src_dir.parent().unwrap_or(&src_dir);
        match extension {
            "rs" => index_rust_files(
                &src_dir, base, component, chunker, chunker_config, model_hash,
                reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
            ),
            "py" => index_python_files(
                &src_dir, base, component, chunker, chunker_config, model_hash,
                reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
            ),
            _ => {}
        }
    }

    // Index specification docs (P1)
    if include_specs {
        let specs_dir = path.join("docs/specifications");
        if specs_dir.exists() {
            index_markdown_files(
                &specs_dir, component, chunker, chunker_config, model_hash,
                reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
            );
        }
    }

    // Index mdBook documentation (P1)
    if include_book {
        let book_dir = path.join("book/src");
        if book_dir.exists() {
            index_markdown_files_recursive(
                &book_dir, book_dir.parent().unwrap_or(&book_dir), component,
                chunker, chunker_config, model_hash,
                reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
            );
        }
    }
}

/// Index all components in a list of directories, optionally printing "not found" for missing dirs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_dir_group(
    dirs: &[&str],
    show_not_found: bool,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    extension: &str,
    include_specs: bool,
    include_book: bool,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    for dir in dirs {
        let path = std::path::Path::new(dir);
        if !path.exists() {
            if show_not_found {
                println!("  {} {} (not found)", "⊘".dimmed(), dir.dimmed());
            }
            continue;
        }
        let component = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        index_component(
            path, component, chunker, chunker_config, model_hash,
            extension, include_specs, include_book,
            reindexer, retriever, indexed_count, total_chunks, fingerprints, chunk_contents,
        );
    }
}

/// Process a single source file for RAG indexing: read, fingerprint, chunk, and print.
#[allow(clippy::too_many_arguments)]
fn process_and_index_file(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    doc_id_prefix: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    retriever: &mut oracle::rag::HybridRetriever,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    if is_trivial_content(&content) {
        return;
    }

    let relative_path = path.strip_prefix(base_dir).unwrap_or(path);
    let doc_id = format!("{}/{}", doc_id_prefix, relative_path.display());

    fingerprints.insert(
        doc_id.clone(),
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
    );

    reindexer.enqueue(&doc_id, path.to_path_buf(), 0);

    let chunk_count = index_file_chunks(
        &content,
        &doc_id,
        chunker,
        retriever,
        total_chunks,
        chunk_contents,
    );
    *indexed_count += 1;

    print_file_indexed(relative_path, chunk_count);
}

/// Check if a single file has changed compared to existing fingerprints
fn check_file_changed(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> Option<bool> {
    use oracle::rag::fingerprint::DocumentFingerprint;

    let content = std::fs::read_to_string(path).ok()?;
    if is_trivial_content(&content) {
        return Some(false);
    }

    let relative_path = path.strip_prefix(base_dir).unwrap_or(path);
    let doc_id = format!("{}/{}", component, relative_path.display());
    let current_fp = DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash);

    match existing_fingerprints.get(&doc_id) {
        Some(stored_fp) if !stored_fp.needs_reindex(&current_fp) => Some(false),
        _ => Some(true), // Changed or new file
    }
}

/// Check if a single entry (file or directory) has changes.
fn entry_has_changes(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    if path.is_dir() && should_recurse_dir(path, should_skip_directory) {
        return check_dir_for_changes(
            path,
            base_dir,
            component,
            chunker_config,
            model_hash,
            existing_fingerprints,
            extension,
        );
    }
    !path.is_dir()
        && path.extension().is_some_and(|ext| ext == extension)
        && check_file_changed(path, base_dir, component, chunker_config, model_hash, existing_fingerprints)
            == Some(true)
}

pub(crate) fn check_dir_for_changes(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };

    entries.flatten().any(|entry| {
        entry_has_changes(
            &entry.path(),
            base_dir,
            component,
            chunker_config,
            model_hash,
            existing_fingerprints,
            extension,
        )
    })
}

/// Recursively index Python files in a directory
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_python_files(
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
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_python_dir) {
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
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "py") {
            process_and_index_file(
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
                chunk_contents,
            );
        }
    }
}

/// Recursively index Rust source files in a directory
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_rust_files(
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
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_rust_dir) {
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
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            process_and_index_file(
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
                chunk_contents,
            );
        }
    }
}

/// Index markdown files in a directory (non-recursive)
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_markdown_files(
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
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file.md");
        let doc_id = format!("{}/docs/{}", component, file_name);

        fingerprints.insert(
            doc_id.clone(),
            oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
        );

        reindexer.enqueue(&doc_id, path.clone(), 0);

        let chunk_count =
            index_file_chunks(&content, &doc_id, chunker, retriever, total_chunks, chunk_contents);
        *indexed_count += 1;

        let display_path = format!("{}/docs/{}", component, file_name);
        print_markdown_indexed(&display_path, chunk_count);
    }
}

/// Recursively index markdown files in a directory (for mdBook)
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_markdown_files_recursive(
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
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_hidden_dir) {
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
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "md") {
            let book_prefix = format!("{}/book", component);
            process_and_index_file(
                &path,
                base_dir,
                &book_prefix,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                retriever,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }
}
