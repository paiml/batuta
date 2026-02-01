//! RAG Indexing Functions
//!
//! This module contains file indexing helpers for the RAG system.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

pub(crate) fn check_dir_for_changes(
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
                        chunk_contents,
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
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
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
                        chunk_contents,
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
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
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
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
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
                        chunk_contents,
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
                    chunk_contents
                        .insert(chunk_id.clone(), chunk.content.chars().take(200).collect());
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
